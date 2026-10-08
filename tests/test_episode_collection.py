"""CPU regressions for the real trainer and state-only vector wrapper.

Run from the repository root with:
    PYTHONPATH=tdmpc2 python -m unittest discover -s tests -v

Requires torch, tensordict, numpy, gymnasium==0.29.1, termcolor and tqdm.
The unused RGB embedding import is isolated so no vision weights or other
MMBench environment dependencies are needed. The real train loop and to_td
run normally; policy evaluation rollouts are skipped.
"""

import contextlib
import importlib.util
import io
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from trainer import Trainer


class Config(SimpleNamespace):

    def get(self, key, default=None):
        return getattr(self, key, default)


def load_vector_wrapper():
    # Load the actual wrapper without envs/__init__.py eagerly importing all
    # optional simulators. The unused RGB dependency must never be constructed.
    embedding = ModuleType('envs.wrappers.embedding')
    def unused_embedding(*args, **kwargs):
        raise AssertionError('This regression exercises state observations only.')
    embedding.EmbeddingWrapper = unused_embedding
    source = Path(__file__).resolve().parents[1] / 'tdmpc2/envs/wrappers/vectorized_multitask.py'
    spec = importlib.util.spec_from_file_location('_collection_vector_wrapper', source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'envs.wrappers.embedding': embedding}):
        spec.loader.exec_module(module)
    return module.VectorizedMultitaskWrapper


VectorizedMultitaskWrapper = load_vector_wrapper()


class CountingEnv(gym.Env):
    """Distinct observations identify environment, episode and decision step.

    Record the actual action and emitted observation/reward sequence before
    Gymnasium's same-step autoreset, independently of the trainer's storage.
    """

    def __init__(self, env_index, length, completed):
        self.env_index = env_index
        self.length = length
        self.completed = completed
        self.episode = -1
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def observation(self):
        return np.array([1000 * self.env_index + 100 * self.episode + self.time], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.time = 0
        obs = self.observation()
        self.observations = [obs.copy()]
        self.actions = [np.full((1,), np.nan, dtype=np.float32)]
        self.rewards = [np.nan]
        return obs, {'success': 0.0, 'score': 0.0}

    def step(self, action):
        self.time += 1
        obs = self.observation()
        reward = float(10 * self.env_index + self.time)
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        truncated = self.time == self.length
        if truncated:
            # The real VecWrapper pads scalar state observations to 128 values.
            expected_obs = torch.zeros(self.length + 1, 128)
            expected_obs[:, :1] = torch.from_numpy(np.stack(self.observations))
            expected_action = torch.zeros(self.length + 1, 16)
            expected_action[0] = float('nan')
            expected_action[:, :1] = torch.from_numpy(np.stack(self.actions))
            self.completed.append({
                'obs': expected_obs,
                'action': expected_action,
                'reward': torch.tensor(self.rewards, dtype=torch.float32),
                'terminated': torch.zeros(self.length + 1, dtype=torch.bool),
                'task': torch.full((self.length + 1,), self.env_index, dtype=torch.int32),
            })
        return obs, reward, False, truncated, {'success': 0.0, 'score': float(self.time)}


class RecordingAgent:
    model = 'CPU collection regression agent'

    def __call__(self, obs, **kwargs):
        # Legal continuous actions vary with each episode's decision step.
        action = torch.zeros(obs.shape[0], 16)
        action[:, :1] = obs[:, :1].remainder(100) / 10
        return action


class RecordingBuffer:

    def __init__(self):
        self.episodes = []

    def add(self, td, world_size=1, rank=0):
        # Replay consumes the episode immediately. Preserve it independently
        # from the trainer's reused cache just as replay.extend copies storage.
        self.episodes.append(td.clone()[0])


class CollectionTests(unittest.TestCase):

    def run_collection(self, lengths, periodic_eval=False):
        completed = []
        cfg = Config(
            num_envs=len(lengths), num_tasks=len(lengths), num_global_tasks=len(lengths),
            tasks=[str(i) for i in range(len(lengths))], task='soup', seed=0,
            episode_lengths=list(lengths), episode_length=max(lengths),
            action_dims=[1] * len(lengths), rank=0, world_size=1, obs='state',
            save_video=False, use_demos=False, checkpoint=None, finetune=True,
            seeding_coef=1000, steps=2 * max(lengths) * len(lengths) - 1,
            eval_freq=max(lengths) * len(lengths) if periodic_eval else 1000000,
            save_freq=1000000, task_embeddings=[],
        )
        buffer = RecordingBuffer()
        logger = SimpleNamespace(log=lambda *args: None, pprint_multitask=lambda *args: None,
                                 finish=lambda: None)
        make_env = lambda child_cfg: CountingEnv(int(child_cfg.task),
                                               lengths[int(child_cfg.task)], completed)
        with contextlib.redirect_stdout(io.StringIO()):
            env = VectorizedMultitaskWrapper(cfg, make_env)
            try:
                trainer = Trainer(cfg, env, RecordingAgent(), buffer, logger)
                # Skip evaluation policy rollouts, retaining the actual train
                # loop's evaluation-boundary reset, constructor and to_td.
                with patch.object(trainer, 'eval', return_value={}):
                    trainer.train()
            finally:
                env.close()

        self.assertGreaterEqual(len(completed), 2 * len(lengths))
        self.assertEqual(len(buffer.episodes), len(completed))
        for episode_index, (actual, expected) in enumerate(zip(buffer.episodes, completed)):
            for key, values in expected.items():
                with self.subTest(episode=episode_index, field=key, lengths=lengths):
                    torch.testing.assert_close(actual[key], values, rtol=0, atol=0, equal_nan=True)

    def test_mixed_lengths(self):
        self.run_collection((3, 6))

    def test_reverse_mixed_lengths(self):
        self.run_collection((6, 3))

    def test_nondivisible_lengths(self):
        self.run_collection((4, 6))

    def test_equal_lengths(self):
        self.run_collection((4, 4))

    def test_single_environment(self):
        self.run_collection((4,))

    def test_multiple_simultaneous_completions(self):
        self.run_collection((3, 6, 6))

    def test_periodic_evaluation_reset(self):
        self.run_collection((3, 6), periodic_eval=True)


if __name__ == '__main__':
    unittest.main()
