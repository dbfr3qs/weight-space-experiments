"""PPO baseline for stage 3: the same 400 -> 32 -> 32 -> 6 tanh policy, trained on CPU
for a fixed wall-clock budget (separate value net of the same shape), then evaluated with
the hand-written policy's protocol (1,000 episodes, sampled actions, sticky 0.1, cap 10k).

  python stage3_ppo.py --minutes 60
"""
import json, sys, time
import numpy as np
import torch
import torch.nn as nn
from minatar import Environment
from stage3_breakout import Policy, evaluate


def main(minutes=60.0, seed=0, n_env=16, n_steps=128, epochs=4, n_mb=4, lr=5e-4,
         gamma=0.99, lam=0.95, clip=0.2, ent=0.01, cap=10_000):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(2)
    pi, vf = Policy(), Policy(n_act=1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(vf.parameters()), lr=lr, eps=1e-5)
    envs = [Environment("breakout", sticky_action_prob=0.1) for _ in range(n_env)]
    for i, e in enumerate(envs): e.seed(seed * 1000 + i); e.reset()
    obs = np.stack([e.state().reshape(-1) for e in envs]).astype(np.float32)
    ep_ret, ep_len, finished = np.zeros(n_env), np.zeros(n_env, int), []
    t0, steps, it, log = time.time(), 0, 0, []
    while time.time() - t0 < minutes * 60:
        O = np.zeros((n_steps, n_env, 400), np.float32); A = np.zeros((n_steps, n_env), np.int64)
        LP = np.zeros((n_steps, n_env), np.float32); Rw = np.zeros((n_steps, n_env), np.float32)
        D = np.zeros((n_steps, n_env), np.float32); V = np.zeros((n_steps + 1, n_env), np.float32)
        for t in range(n_steps):
            with torch.no_grad():
                ot = torch.from_numpy(obs)
                dist = torch.distributions.Categorical(logits=pi(ot))
                a = dist.sample(); V[t] = vf(ot).squeeze(-1).numpy()
            O[t], A[t], LP[t] = obs, a.numpy(), dist.log_prob(a).numpy()
            for i, e in enumerate(envs):
                r, done = e.act(int(A[t, i]))
                ep_ret[i] += r; ep_len[i] += 1; Rw[t, i] = r
                if done or ep_len[i] >= cap:
                    D[t, i] = 1.0; finished.append(ep_ret[i]); ep_ret[i] = 0; ep_len[i] = 0; e.reset()
                obs[i] = e.state().reshape(-1)
        with torch.no_grad():
            V[n_steps] = vf(torch.from_numpy(obs)).squeeze(-1).numpy()
        adv = np.zeros_like(Rw); g = 0
        for t in reversed(range(n_steps)):
            delta = Rw[t] + gamma * V[t + 1] * (1 - D[t]) - V[t]
            g = delta + gamma * lam * (1 - D[t]) * g; adv[t] = g
        ret = adv + V[:-1]
        bO, bA, bLP = map(lambda z: torch.from_numpy(z.reshape(n_steps * n_env, *z.shape[2:])), (O, A, LP))
        bAdv, bRet = torch.from_numpy(adv.reshape(-1)), torch.from_numpy(ret.reshape(-1))
        frac = 1 - (time.time() - t0) / (minutes * 60)
        for gopt in opt.param_groups: gopt["lr"] = lr * max(frac, 0.05)
        N = n_steps * n_env
        for _ in range(epochs):
            perm = torch.randperm(N)
            for mb in perm.split(N // n_mb):
                d = torch.distributions.Categorical(logits=pi(bO[mb]))
                ratio = (d.log_prob(bA[mb]) - bLP[mb]).exp()
                ad = bAdv[mb]; ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pl = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
                vl = ((vf(bO[mb]).squeeze(-1) - bRet[mb]) ** 2).mean()
                loss = pl + 0.5 * vl - ent * d.entropy().mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(pi.parameters()) + list(vf.parameters()), 0.5); opt.step()
        steps += N; it += 1
        if it % 20 == 0:
            recent = float(np.mean(finished[-100:])) if finished else float("nan")
            log.append((steps, recent)); print(f"steps {steps}  recent-100 return {recent:.2f}  "
                                               f"{steps / (time.time() - t0):.0f} steps/s", flush=True)
    torch.save(pi.state_dict(), f"ppo_breakout_s{seed}.pt")
    res = evaluate(pi, episodes=1000)
    res.update(train_steps=steps, train_minutes=minutes, seed=seed)
    print("ppo:", res, flush=True)
    json.dump(dict(ppo=res, curve=log), open(f"results_stage3_ppo_s{seed}.json", "w"), indent=1)


if __name__ == "__main__":
    mins = float(sys.argv[sys.argv.index("--minutes") + 1]) if "--minutes" in sys.argv else 60.0
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    main(mins, seed)
