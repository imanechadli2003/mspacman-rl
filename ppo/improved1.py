"""
PPO Amélioré 1 - Ms. Pac-Man
Groupe 7 : chai0761, lepv0744, robm8964, trej2436

Amélioration : Observation enrichie avec une carte de distance aux bonbons.

Idée :
  Au lieu de donner seulement les pixels bruts à l'agent, on ajoute un
  5e canal à l'observation : une carte de distance (distance map).
  Pour chaque pixel, cette carte indique à quelle distance se trouve
  le bonbon le plus proche. Plus la valeur est grande, plus on est loin.

  L'agent n'a plus à "deviner" où sont les bonbons à partir des pixels —
  l'information est fournie explicitement. Cela devrait accélérer
  l'apprentissage et aider l'agent à mieux planifier ses déplacements.

Implémentation :
  On crée un wrapper gymnasium qui :
    1. Intercepte le frame RGB avant grayscale
    2. Détecte les bonbons par leur couleur dans l'espace HSV
    3. Calcule la distance transform (scipy) : chaque pixel reçoit
       sa distance au bonbon le plus proche
    4. Normalise la carte entre 0 et 255
    5. La concatène aux 4 frames grayscale empilées → shape (5, 84, 84)

  Corrections par rapport à la version précédente :
    - Ajout du wrapper Monitor (requis pour que info["episode"] soit rempli)
    - Observation en format channel-first (C, H, W) attendu par CnnPolicy
    - Suppression du VecTransposeImage redondant
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py
from scipy.ndimage import distance_transform_edt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import AtariPreprocessing

# ─────────────────────────────────────────────
# 1. PARAMÈTRES
# ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
N_ENVS          = 1
N_STACK         = 4
SAVE_PATH       = "./results/ppo_improved1"
LOG_PATH        = "./results/logs/ppo_improved1"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


# ─────────────────────────────────────────────
# 2. WRAPPER : carte de distance aux bonbons
# ─────────────────────────────────────────────
class PelletDistanceMapWrapper(gym.ObservationWrapper):
    """
    Ajoute un 2e canal à chaque frame : la carte de distance aux bonbons.

    L'observation de base après AtariPreprocessing est (1, 84, 84)
    en format channel-first (C, H, W). On ajoute notre canal distance
    pour obtenir (2, 84, 84) par frame. Après VecFrameStack(n=4),
    l'observation finale sera (8, 84, 84) : 4 × 2 canaux.

    Chaque pixel du canal distance contient la distance euclidienne
    normalisée (0-255) au bonbon le plus proche. Les zones proches
    des bonbons ont une valeur haute (utile pour l'agent).

    Détection des bonbons :
      Les bonbons MsPacman sont jaune pâle, les power pellets sont blancs.
      On les détecte via deux plages HSV combinées.

    Distance transform :
      scipy.ndimage.distance_transform_edt calcule pour chaque pixel
      à 0 (pas de bonbon) la distance au pixel à 1 le plus proche.
      On inverse pour que les zones proches des bonbons valent 255.
    """

    PELLET_HSV_LOW   = np.array([15,  80, 150], dtype=np.uint8)
    PELLET_HSV_HIGH  = np.array([45, 255, 255], dtype=np.uint8)
    POWERUP_HSV_LOW  = np.array([0,   0,  200], dtype=np.uint8)
    POWERUP_HSV_HIGH = np.array([180, 40, 255], dtype=np.uint8)

    def __init__(self, env):
        super().__init__(env)

        # AtariPreprocessing avec grayscale_newaxis=False donne (84, 84)
        # On va travailler en channel-first : (1, 84, 84) puis (2, 84, 84)
        old_shape = self.observation_space.shape   # (84, 84) sans newaxis

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(2, old_shape[0], old_shape[1]),   # (2, 84, 84)
            dtype=np.uint8
        )
        self._last_rgb = None

    def observation(self, obs):
        # obs : (84, 84) uint8 — frame grayscale sans dimension de canal
        gray_channel = obs[np.newaxis, :, :]          # (1, 84, 84)
        dist_map     = self._build_distance_map()     # (84, 84)
        dist_channel = dist_map[np.newaxis, :, :]     # (1, 84, 84)
        return np.concatenate([gray_channel, dist_channel], axis=0)  # (2, 84, 84)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            self._last_rgb = self.env.render()
        except Exception:
            self._last_rgb = None
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        try:
            self._last_rgb = self.env.render()
        except Exception:
            self._last_rgb = None
        return self.observation(obs), info

    def _build_distance_map(self):
        """
        Construit la carte de distance aux bonbons (84×84, uint8).

        Étapes :
          1. Détecter les bonbons dans le frame RGB via masque HSV
          2. Calculer la distance transform euclidienne
          3. Inverser et normaliser → zones proches des bonbons = 255
        """
        if self._last_rgb is None:
            return np.zeros((84, 84), dtype=np.uint8)

        hsv  = cv2.cvtColor(self._last_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, self.PELLET_HSV_LOW,  self.PELLET_HSV_HIGH),
            cv2.inRange(hsv, self.POWERUP_HSV_LOW, self.POWERUP_HSV_HIGH)
        )

        if mask.sum() == 0:
            return np.zeros((84, 84), dtype=np.uint8)

        mask_small = cv2.resize(mask, (84, 84), interpolation=cv2.INTER_NEAREST)
        pellet_map = (mask_small > 0).astype(np.uint8)

        binary_inv = 1 - pellet_map
        dist       = distance_transform_edt(binary_inv).astype(np.float32)

        max_dist = dist.max()
        if max_dist > 0:
            inverted = (1.0 - dist / max_dist) * 255
        else:
            inverted = np.zeros_like(dist)

        return inverted.astype(np.uint8)


# ─────────────────────────────────────────────
# 3. CALLBACK : enregistre le score à chaque épisode
# ─────────────────────────────────────────────
class ScoreCallback(BaseCallback):
    """
    À chaque fin d'épisode, on récupère le score obtenu
    et on le sauvegarde pour faire un graphique ensuite.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                print(f"  Épisode {len(self.episode_rewards):4d} | "
                      f"Score : {info['episode']['r']:7.1f} | "
                      f"Steps : {info['episode']['l']:5d}")
        return True


# ─────────────────────────────────────────────
# 4. CRÉATION DE L'ENVIRONNEMENT
# ─────────────────────────────────────────────
gym.register_envs(ale_py)

print("Création de l'environnement avec carte de distance aux bonbons...")

def make_env_with_distance_map(seed=42):
    """
    Crée un env MsPacman avec le wrapper de carte de distance.

    Stack des wrappers :
        gym.make → AtariPreprocessing → PelletDistanceMapWrapper → Monitor
    Le Monitor est INDISPENSABLE : c'est lui qui injecte info["episode"]
    à la fin de chaque épisode, ce que lit le ScoreCallback.
    """
    env = gym.make("ALE/MsPacman-v5", render_mode="rgb_array")
    env.reset(seed=seed)

    # Preprocessing Atari standard — grayscale_newaxis=False → obs (84, 84)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,
        screen_size=84,
        grayscale_obs=True,
        grayscale_newaxis=False,  # on gère le canal nous-mêmes dans le wrapper
        scale_obs=False,
    )

    # Notre wrapper ajoute le canal distance → obs (2, 84, 84)
    env = PelletDistanceMapWrapper(env)

    # Monitor est requis pour que info["episode"] soit rempli
    env = Monitor(env)
    return env

vec_env = DummyVecEnv([lambda: make_env_with_distance_map(seed=42)])

# VecFrameStack empile les 4 dernières obs : (2, 84, 84) × 4 → (8, 84, 84)
vec_env = VecFrameStack(vec_env, n_stack=N_STACK)

print(f"Observation shape : {vec_env.observation_space.shape}")  # (8, 84, 84)
print(f"Action space      : {vec_env.action_space}")


# ─────────────────────────────────────────────
# 5. CRÉATION DU MODÈLE PPO
# ─────────────────────────────────────────────
print("\nCréation du modèle PPO avec carte de distance...")

model = PPO(
    policy="CnnPolicy",
    env=vec_env,
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    verbose=0,
    tensorboard_log=LOG_PATH,
    seed=42,
    device="cpu"
)

print("Modèle créé !")
print(f"Politique : {model.policy}")


# ─────────────────────────────────────────────
# 6. ENTRAÎNEMENT
# ─────────────────────────────────────────────
callback = ScoreCallback()

print(f"\nDébut de l'entraînement ({TOTAL_TIMESTEPS:,} steps)...")
print("─" * 55)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback,
    progress_bar=True
)

print("─" * 55)
print("Entraînement terminé !")


# ─────────────────────────────────────────────
# 7. SAUVEGARDE DU MODÈLE ET DES RÉSULTATS
# ─────────────────────────────────────────────
model.save(f"{SAVE_PATH}/model")
print(f"\nModèle sauvegardé dans {SAVE_PATH}/model.zip")

scores_path = f"{LOG_PATH}/scores.csv"
with open(scores_path, "w") as f:
    f.write("episode,score,length\n")
    for i, (r, l) in enumerate(zip(callback.episode_rewards, callback.episode_lengths)):
        f.write(f"{i+1},{r},{l}\n")
print(f"Scores sauvegardés dans {scores_path}")


# ─────────────────────────────────────────────
# 8. GRAPHIQUE DES RÉSULTATS
# ─────────────────────────────────────────────
if len(callback.episode_rewards) > 0:
    rewards = np.array(callback.episode_rewards)

    window   = min(10, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color="green", label="Score brut")
    plt.plot(range(window-1, len(rewards)), smoothed,
             color="green", linewidth=2, label=f"Moyenne glissante ({window} épisodes)")
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.title("PPO Amélioré 1 — Carte de distance aux bonbons (Ms. Pac-Man)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = f"{SAVE_PATH}/scores_improved1.png"
    plt.savefig(graph_path)
    print(f"Graphique sauvegardé dans {graph_path}")
    plt.close()

    print(f"\nRésumé :")
    print(f"  Nombre d'épisodes : {len(rewards)}")
    print(f"  Score moyen       : {rewards.mean():.1f}")
    print(f"  Score max         : {rewards.max():.1f}")
    print(f"  Score min         : {rewards.min():.1f}")
else:
    print("\nAucun épisode complet enregistré.")

vec_env.close()
print("\nDone !")
