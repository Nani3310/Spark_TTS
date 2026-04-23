import json
import os
import random
import re
from datetime import datetime

GENDERS = ["male", "female"]
PITCHES = [1, 2, 3, 4, 5]
SPEEDS  = [1, 2, 3, 4, 5]

# Build a flat list of all (gender, pitch, speed) tuples
ALL_ACTIONS = [
    (g, p, s)
    for g in GENDERS
    for p in PITCHES
    for s in SPEEDS
]
# 2 × 5 × 5 = 50 actions
N_ACTIONS = len(ALL_ACTIONS)

# Text length buckets
LENGTH_SHORT  = "short"   # < 50 chars
LENGTH_MEDIUM = "medium"  # 50–150 chars
LENGTH_LONG   = "long"    # > 150 chars

# Sentiment buckets (keyword-based, no model needed)
SENT_POSITIVE = "positive"
SENT_NEGATIVE = "negative"
SENT_NEUTRAL  = "neutral"

POSITIVE_WORDS = {
    "happy", "great", "wonderful", "love", "amazing", "excellent",
    "good", "fantastic", "joy", "excited", "congratulations", "thanks",
    "beautiful", "awesome", "brilliant", "perfect", "celebrate",
}
NEGATIVE_WORDS = {
    "sad", "sorry", "terrible", "hate", "awful", "bad", "angry",
    "unfortunately", "failed", "error", "problem", "wrong", "died",
    "horrible", "disaster", "fear", "pain", "cry", "loss",
}

# All (length, sentiment) combos = 3 × 3 = 9 states
ALL_STATES = [
    f"{l}_{s}"
    for l in [LENGTH_SHORT, LENGTH_MEDIUM, LENGTH_LONG]
    for s in [SENT_POSITIVE, SENT_NEGATIVE, SENT_NEUTRAL]
]
N_STATES = len(ALL_STATES)


ALPHA   = 0.1    # learning rate
GAMMA   = 0.9    # discount factor (not used in bandit mode, kept for extension)
EPSILON = 0.15   # exploration rate (15% random, 85% greedy)
Q_TABLE_PATH = "rl_qtable.json"

# HELPER FUNCTIONS

def get_text_state(text: str) -> str:
    """Convert raw text into a state string."""
    text = text.strip()

    # Length
    if len(text) < 50:
        length = LENGTH_SHORT
    elif len(text) <= 150:
        length = LENGTH_MEDIUM
    else:
        length = LENGTH_LONG

    # Sentiment (simple keyword count)
    words   = set(re.findall(r"\b\w+\b", text.lower()))
    pos_hit = len(words & POSITIVE_WORDS)
    neg_hit = len(words & NEGATIVE_WORDS)

    if pos_hit > neg_hit:
        sentiment = SENT_POSITIVE
    elif neg_hit > pos_hit:
        sentiment = SENT_NEGATIVE
    else:
        sentiment = SENT_NEUTRAL

    return f"{length}_{sentiment}"


def normalise_reward(stars: int) -> float:
    """Convert 1–5 star rating to [-1, +1]."""
    return (stars - 3) / 2.0   # 1→-1, 3→0, 5→+1


def action_to_str(action: tuple) -> str:
    return f"{action[0]}_{action[1]}_{action[2]}"


def str_to_action(s: str) -> tuple:
    parts = s.split("_")
    return (parts[0], int(parts[1]), int(parts[2]))

# Q-TABLE  (state → action → Q-value)

def _empty_qtable() -> dict:
    return {
        state: {action_to_str(a): 0.0 for a in ALL_ACTIONS}
        for state in ALL_STATES
    }


def load_qtable() -> dict:
    if os.path.exists(Q_TABLE_PATH):
        try:
            with open(Q_TABLE_PATH) as f:
                qt = json.load(f)
            print(f"[RL] Q-table loaded from {Q_TABLE_PATH} ✅")
            return qt
        except Exception as e:
            print(f"[RL] Could not load Q-table: {e} — starting fresh")
    return _empty_qtable()


def save_qtable(qt: dict):
    try:
        with open(Q_TABLE_PATH, "w") as f:
            json.dump(qt, f, indent=2)
    except Exception as e:
        print(f"[RL] Could not save Q-table: {e}")



class TTSRLAgent:

    def __init__(self):
        self.qt           = load_qtable()
        self.history      = []   # list of episode dicts for logging
        self.episode_count = 0
        print(f"[RL] Agent ready — {N_STATES} states × {N_ACTIONS} actions")

    def suggest(self, text: str) -> tuple:
        """
        Return (gender, pitch, speed) for the given text.
        Uses epsilon-greedy: explores randomly 15% of the time,
        otherwise picks the action with highest Q-value.
        """
        state = get_text_state(text)

        if state not in self.qt:
            self.qt[state] = {action_to_str(a): 0.0 for a in ALL_ACTIONS}

        if random.random() < EPSILON:
            # EXPLORE — random action
            action = random.choice(ALL_ACTIONS)
            mode   = "explore"
        else:
            # EXPLOIT — best known action
            q_row  = self.qt[state]
            best_a = max(q_row, key=q_row.get)
            action = str_to_action(best_a)
            mode   = "exploit"

        gender, pitch, speed = action
        print(f"[RL] State='{state}' → {mode} → gender={gender} pitch={pitch} speed={speed}")
        return gender, pitch, speed

    def update(self, text: str, gender: str, pitch: int, speed: int, stars: int):
        
        state  = get_text_state(text)
        action = action_to_str((gender, pitch, speed))
        reward = normalise_reward(stars)

        if state not in self.qt:
            self.qt[state] = {action_to_str(a): 0.0 for a in ALL_ACTIONS}

        # Q-learning update rule:
        # Q(s,a) ← Q(s,a) + α × (reward - Q(s,a))
        old_q = self.qt[state].get(action, 0.0)
        new_q = old_q + ALPHA * (reward - old_q)
        self.qt[state][action] = new_q

        # Log this episode
        self.episode_count += 1
        self.history.append({
            "episode"   : self.episode_count,
            "timestamp" : datetime.now().isoformat(),
            "state"     : state,
            "action"    : action,
            "stars"     : stars,
            "reward"    : round(reward, 3),
            "old_q"     : round(old_q, 4),
            "new_q"     : round(new_q, 4),
        })

        save_qtable(self.qt)
        print(f"[RL] Updated Q({state}, {action}): {old_q:.4f} → {new_q:.4f} (reward={reward:+.2f})")

    def get_best_params(self, text: str) -> tuple:
        
        state = get_text_state(text)
        if state not in self.qt:
            return "male", 3, 3
        q_row  = self.qt[state]
        best_a = max(q_row, key=q_row.get)
        return str_to_action(best_a)

    def get_stats(self) -> str:
        
        if not self.history:
            return "No feedback received yet. Rate some outputs to train the agent!"

        lines = [f"📊 RL Agent Stats — {self.episode_count} episodes\n"]
        lines.append(f"{'State':<25} {'Best Action':<25} {'Q-value':>8}")
        lines.append("─" * 62)

        for state in ALL_STATES:
            if state not in self.qt:
                continue
            q_row  = self.qt[state]
            best_a = max(q_row, key=q_row.get)
            best_q = q_row[best_a]
            if best_q != 0.0:   # only show states that have been trained
                lines.append(f"{state:<25} {best_a:<25} {best_q:>+8.4f}")

        if len(lines) == 3:
            lines.append("  (no states trained yet)")

        avg_reward = sum(e["reward"] for e in self.history) / len(self.history)
        lines.append(f"\nAverage reward : {avg_reward:+.3f}")
        lines.append(f"Exploration ε  : {EPSILON}")
        lines.append(f"Learning rate α: {ALPHA}")
        return "\n".join(lines)

    def get_history_csv(self) -> str:
        """Return episode history as CSV string."""
        if not self.history:
            return "episode,timestamp,state,action,stars,reward,old_q,new_q"
        keys = list(self.history[0].keys())
        rows = [",".join(keys)]
        for ep in self.history:
            rows.append(",".join(str(ep[k]) for k in keys))
        return "\n".join(rows)



if __name__ == "__main__":
    agent = TTSRLAgent()

    texts = [
        "Hello, I am happy to meet you!",
        "I am very sorry for your loss.",
        "The quarterly financial report shows significant growth in revenue across all sectors.",
    ]

    for text in texts:
        print(f"\nText: '{text[:50]}'")
        g, p, s = agent.suggest(text)
        print(f"  Suggested: gender={g} pitch={p} speed={s}")
        # simulate user rating
        agent.update(text, g, p, s, stars=4)

    print("\n" + agent.get_stats())