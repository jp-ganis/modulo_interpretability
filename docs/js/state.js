// A tiny observable store: the single source of truth that every panel and
// control reads from. Anything that wants to change the view calls
// store.set({...}); every subscriber is told what changed and redraws only
// what it needs. This is what keeps the panels in sync with each other.

export function createStore(initial) {
  let state = { ...initial };
  const subscribers = new Set();

  return {
    get: () => state,
    set(patch) {
      const prev = state;
      state = { ...state, ...patch };
      for (const fn of subscribers) fn(state, prev);
    },
    subscribe(fn) {
      subscribers.add(fn);
      return () => subscribers.delete(fn);
    },
  };
}

// True if any of `keys` differs between two states.
export function changed(state, prev, ...keys) {
  return keys.some((k) => state[k] !== prev[k]);
}
