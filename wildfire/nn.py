"""A minimal reverse-mode autodiff on NumPy arrays (manual gradients), plus Adam.

Only the operations the models need are implemented. Gradients are checked
against finite differences in tests/test_nn.py."""

from __future__ import annotations

import numpy as np


class T:
    """A node in the computation graph."""

    __slots__ = ("v", "g", "parents", "back")

    def __init__(self, v, parents=(), back=None):
        self.v = v
        self.g = None
        self.parents = parents
        self.back = back

    def __add__(self, o):
        return add(self, o)

    def __mul__(self, o):
        return mul(self, o)

    def __matmul__(self, o):
        return matmul(self, o)


def const(x):
    return T(np.asarray(x))


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1:
            g = g.sum(i, keepdims=True)
    return g


def add(a, b):
    def back(g):
        return _unbroadcast(g, a.v.shape), _unbroadcast(g, b.v.shape)

    return T(a.v + b.v, (a, b), back)


def sub(a, b):
    def back(g):
        return _unbroadcast(g, a.v.shape), -_unbroadcast(g, b.v.shape)

    return T(a.v - b.v, (a, b), back)


def mul(a, b):
    def back(g):
        return _unbroadcast(g * b.v, a.v.shape), _unbroadcast(g * a.v, b.v.shape)

    return T(a.v * b.v, (a, b), back)


def matmul(a, b):
    def back(g):
        if b.v.ndim == 1:
            return np.multiply.outer(g, b.v), a.v.reshape(-1, a.v.shape[-1]).T @ g.reshape(-1)
        ga = g @ b.v.T
        gb = a.v.reshape(-1, a.v.shape[-1]).T @ g.reshape(-1, g.shape[-1])
        return ga, gb

    return T(a.v @ b.v, (a, b), back)


def relu(a):
    m = a.v > 0

    def back(g):
        return (g * m,)

    return T(a.v * m, (a,), back)


def sigmoid(a):
    s = 0.5 * (1.0 + np.tanh(0.5 * a.v))

    def back(g):
        return (g * s * (1 - s),)

    return T(s, (a,), back)


def concat(ts, axis=-1):
    sizes = [t.v.shape[axis] for t in ts]
    idx = np.cumsum([0] + sizes)

    def back(g):
        return tuple(np.take(g, np.arange(idx[i], idx[i + 1]), axis=axis) for i in range(len(ts)))

    return T(np.concatenate([t.v for t in ts], axis=axis), tuple(ts), back)


def stack(ts, axis=0):
    def back(g):
        return tuple(np.take(g, i, axis=axis) for i in range(len(ts)))

    return T(np.stack([t.v for t in ts], axis=axis), tuple(ts), back)


def index(a, key):
    """a.v[key] for basic (non-fancy) indexing."""

    def back(g):
        z = np.zeros_like(a.v)
        z[key] = g
        return (z,)

    return T(a.v[key], (a,), back)


def sum_(a, axis):
    def back(g):
        return (np.broadcast_to(np.expand_dims(g, axis), a.v.shape).copy(),)

    return T(a.v.sum(axis), (a,), back)


def one_minus(a):
    def back(g):
        return (-g,)

    return T(1.0 - a.v, (a,), back)


def bce_with_logits(z, y):
    """Elementwise binary cross-entropy from logits (y constant)."""
    v = np.maximum(z.v, 0) - z.v * y + np.log1p(np.exp(-np.abs(z.v)))
    s = 0.5 * (1.0 + np.tanh(0.5 * z.v))

    def back(g):
        return (g * (s - y),)

    return T(v, (z,), back)


def bce_prob(p, y, eps=1e-6):
    pc = np.clip(p.v, eps, 1 - eps)
    v = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    inside = (p.v > eps) & (p.v < 1 - eps)

    def back(g):
        return (g * inside * (-(y / pc) + (1 - y) / (1 - pc)),)

    return T(v, (p,), back)


def weighted_mean(a, w):
    """sum(a * w) / sum(w), w constant."""
    tot = max(float(w.sum()), 1.0)

    def back(g):
        return (g * w / tot,)

    return T(np.asarray((a.v * w).sum() / tot), (a,), back)


def backward(out):
    """Accumulate gradients of scalar `out` into every node's .g."""
    order = []
    seen = set()
    stack_ = [(out, False)]
    while stack_:
        node, done = stack_.pop()
        if done:
            order.append(node)
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack_.append((node, True))
        for p in node.parents:
            if id(p) not in seen:
                stack_.append((p, False))
    for n in order:
        n.g = None
    out.g = np.ones_like(out.v)
    for n in reversed(order):
        if n.back is None or n.g is None:
            continue
        gs = n.back(n.g)
        for p, g in zip(n.parents, gs, strict=True):
            p.g = g if p.g is None else p.g + g


class Adam:
    def __init__(self, params: dict, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.lr = lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.s = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.s[k] = self.b2 * self.s[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1**self.t)
            sh = self.s[k] / (1 - self.b2**self.t)
            self.p[k] -= (lr * mh / (np.sqrt(sh) + self.eps)).astype(self.p[k].dtype)
