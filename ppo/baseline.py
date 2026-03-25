"""
PPO Baseline - Ms. Pac-Man
Groupe 7 : chai0761, lepv0744, robm8964, trej2436

Ce script entraîne un agent PPO sans aucune modification.
C'est notre point de référence pour comparer avec la version améliorée.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

# ─────────────────────────────────────────────
# 1. PARAMÈTRES
# ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 200_000   # Nombre de steps d'entraînement
N_ENVS          = 1         # Nombre d'environnements parallèles
N_STACK         = 4         # Nombre de frames empilées
SAVE_PATH       = "./results/ppo_baseline"
LOG_PATH        = "./results/logs/ppo_baseline"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)


# ─────────────────────────────────────────────
# 2. CALLBACK : enregistre le score à chaque épisode
# ─────────────────────────────────────────────
class ScoreCallback(BaseCallback):
    """
    À chaque fin d'épisode, on récupère le score obtenu
    et on le sauvegarde pour faire un graphique ensuite.
    """
    def __init__(self):
        super().__init__()
        self.episode_rewards = []   # liste des scores par épisode
        self.episode_lengths = []   # liste des durées par épisode

    def _on_step(self) -> bool:
        # SB3 remplit automatiquement infos avec ep_rew_mean quand un épisode se termine
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                print(f"  Épisode {len(self.episode_rewards):4d} | "
                      f"Score : {info['episode']['r']:7.1f} | "
                      f"Steps : {info['episode']['l']:5d}")
        return True


# ─────────────────────────────────────────────
# 3. CRÉATION DE L'ENVIRONNEMENT
# ─────────────────────────────────────────────
gym.register_envs(ale_py)

print("Création de l'environnement Ms. Pac-Man...")
env = make_atari_env(
    "ALE/MsPacman-v5",
    n_envs=N_ENVS,
    seed=42
)
# On empile 4 frames consécutives pour que l'agent perçoive le mouvement
env = VecFrameStack(env, n_stack=N_STACK)
print(f"Observation shape : {env.observation_space.shape}")
print(f"Action space      : {env.action_space}")


# ─────────────────────────────────────────────
# 4. CRÉATION DU MODÈLE PPO
# ─────────────────────────────────────────────
print("\nCréation du modèle PPO baseline...")
model = PPO(
    policy="CnnPolicy",     # réseau de neurones convolutif (pour les images)
    env=env,
    learning_rate=2.5e-4,   # vitesse d'apprentissage
    n_steps=128,            # steps collectés avant chaque mise à jour
    batch_size=256,         # taille des mini-batchs
    n_epochs=4,             # nombre de passes sur les données collectées
    gamma=0.99,             # facteur de discount (importance du futur)
    gae_lambda=0.95,        # GAE lambda pour l'estimation des avantages
    clip_range=0.1,         # clipping PPO (empêche les mises à jour trop grandes)
    ent_coef=0.01,          # coefficient d'entropie (encourage l'exploration)
    verbose=0,              # pas de logs internes SB3 (on gère nous-mêmes)
    tensorboard_log=LOG_PATH,
    seed=42,
    device="cpu"            # on force le CPU (pas de GPU disponible)
)

print("Modèle créé !")
print(f"Politique : {model.policy}")


# ─────────────────────────────────────────────
# 5. ENTRAÎNEMENT
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
# 6. SAUVEGARDE DU MODÈLE ET DES RÉSULTATS
# ─────────────────────────────────────────────
model.save(f"{SAVE_PATH}/model")
print(f"\nModèle sauvegardé dans {SAVE_PATH}/model.zip")

# Sauvegarder les scores dans un fichier CSV
scores_path = f"{LOG_PATH}/scores.csv"
with open(scores_path, "w") as f:
    f.write("episode,score,length\n")
    for i, (r, l) in enumerate(zip(callback.episode_rewards, callback.episode_lengths)):
        f.write(f"{i+1},{r},{l}\n")
print(f"Scores sauvegardés dans {scores_path}")


# ─────────────────────────────────────────────
# 7. GRAPHIQUE DES RÉSULTATS
# ─────────────────────────────────────────────
if len(callback.episode_rewards) > 0:
    rewards = np.array(callback.episode_rewards)

    # Moyenne glissante sur 10 épisodes pour lisser la courbe
    window = min(10, len(rewards))
    smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, color="blue", label="Score brut")
    plt.plot(range(window-1, len(rewards)), smoothed,
             color="blue", linewidth=2, label=f"Moyenne glissante ({window} épisodes)")
    plt.xlabel("Épisode")
    plt.ylabel("Score")
    plt.title("PPO Baseline — Score par épisode (Ms. Pac-Man)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    graph_path = f"{SAVE_PATH}/scores_baseline.png"
    plt.savefig(graph_path)
    print(f"Graphique sauvegardé dans {graph_path}")
    plt.close()

    print(f"\nRésumé :")
    print(f"  Nombre d'épisodes : {len(rewards)}")
    print(f"  Score moyen       : {rewards.mean():.1f}")
    print(f"  Score max         : {rewards.max():.1f}")
    print(f"  Score min         : {rewards.min():.1f}")

env.close()
print("\nDone !")
