"""MultiAgentBench's database diagnosis in the online swarm (feedback_state.online_swarm): one planner, the same six peers on every
operation, the record read before and written after every planner iteration.

A task (a PostgreSQL database with an injected performance anomaly; tasks in order, `lanes` at once, a lane being one
PostgreSQL cluster) is prepared once for every track: reset, the scenario's schema and benign queries, the anomaly workloads.
Each track then runs the benchmark's star loop with its own planner (the central model as MARBLE's EnginePlanner, over the
task's five agent profiles, one per root cause); a planner iteration is a tick. Every root cause the planner assigns that the
track has not investigated yet is an event, "is X a root cause?":

  investigators  the six peers, each a MARBLE agent with the cause's profile, the planner's assignment, the query_db action (one
                 SQL statement per step) and its own memory, on its own model; on record tracks also the central model itself
                 (its own answer, recorded, never shown). Each ends with its finding ('Final answer: yes|no').
  commit         the central model reads the six findings (tilted on record tracks) and commits the cause's verdict; the planner
                 receives the verdict as that agent's result, so its next assignments and its final decision read it.
  labels         every finding and verdict against the injected anomaly, as soon as it is in: nobody's label depends on what a
                 track committed.

All five causes are investigated on every task. The planner chooses the order: the causes it names that are not done yet, in its
order; an iteration where it names none takes the next cause in the benchmark's order; once it stops (decide_next_step) or its
iterations run out, the causes left are dispatched together. When every cause is done the planner decides from all five
verdicts (the benchmark's summarize_output), scored by the benchmark's rule (a true cause among its allowed guesses), exactly
(the predicted set is the true set) and by set F1; the verdicts are a second diagnosis (the causes the track said yes to).

Injected: prepare(lane, task) -> the injection's log; planner_for(track, lane, task) -> {assign() -> {agent id: assignment},
update(summary), decide(results) -> continue?, final(summary) -> text}; investigate(lane, model, task, agent, assignment) ->
{text, evidence}; peers: the peers' model names in peer order; own_model: the central model's served name.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from feedback_state.online_swarm import Track

TASK_TYPE = "boolqa"


def verdict(text: str) -> str | None:
    """YES / NO / None: a finding's or a verdict's 'Final answer' (feedback_state.swarm.verdict_of), else its last yes/no."""
    from feedback_state.swarm import verdict_of
    from feedback_state.tasks import boolqa_extract_answer

    v = verdict_of(text)
    if v is None:
        b = boolqa_extract_answer(str(text or "").rsplit("</think>", 1)[-1])
        v = b.upper() if b else None
    return v


def grade(ev: dict, text: str) -> int:
    return int(verdict(text) == ("YES" if ev["answer"] == "yes" else "NO"))


def cause_event(task: dict, agent: dict, cause: str, assignment: str, iteration: int, forced: bool) -> dict:
    """The event of one root cause of one task (the pool stream's format, pipeline.swarm merge-pool)."""
    return {"id": f"{task['id']}::{cause}", "task_type": TASK_TYPE, "source": task["scenario"], "task_id": task["id"], "cause": cause,
            "agent_id": agent["agent_id"], "problem": f"{task['task'].strip()}\n\nQuestion: is {cause} a root cause of this database's performance issue?",
            "context": "", "answer": "yes" if cause in task["root_causes"] else "no", "root_causes": list(task["root_causes"]),
            "number_of_labels_pred": int(task["number_of_labels_pred"]), "labels": list(task["labels"]), "assignment": assignment,
            "iteration": int(iteration), "forced": bool(forced), "peer_responses": {}, "peer_correct": {}, "correctness_by_peer": {}}


def _key(agent_id) -> str:
    """An agent id as the planner may write it ('agent1', 'Agent 1', 'agent_1') reduced to one form."""
    return re.sub(r"[^a-z0-9]", "", str(agent_id).lower())


def default_assignment(cause: str) -> str:
    return f"Explore the possibility of {cause} as a root cause of the database's performance issue."


def diagnosis(verdicts: dict[str, str], labels: list[str]) -> str:
    """The verdicts as a diagnosis: the causes said yes to, in the benchmark's order."""
    yes = [c for c in labels if verdict(verdicts.get(c, "")) == "YES"]
    return "Final answer: " + (", ".join(yes) if yes else "NONE")


def scored(text: str, task: dict) -> dict:
    from feedback_state.swarm import graded

    g = graded(text, task)
    pred, truth = set(g["predicted"]), set(task["root_causes"])
    g["f1"] = 0.0 if not pred else 2 * len(pred & truth) / (len(pred) + len(truth))
    return g


@dataclass
class TaskState:
    task: dict
    planner: object
    agent_of: dict                                        # cause -> the task's agent (id, profile)
    remaining: list                                       # causes not investigated yet, in the benchmark's order
    injection: object = None
    iteration: int = 0
    forced: bool = False                                  # the planner stopped or ran out of iterations: the rest go together
    results: list = field(default_factory=list)           # [{agent id: committed verdict}] per cause, for the final decision
    tick: list = field(default_factory=list)              # this tick's, for the planner's progress
    verdicts: dict = field(default_factory=dict)          # cause -> committed verdict
    done: bool = False


class MarbleDBAdapter:
    def __init__(self, tasks: list[dict], *, peers: list[str], own_model: str, prepare, planner_for, investigate, lanes: int = 1,
                 workers: int = 32, log=print):
        self.tasks, self.peer_models, self.own_model = list(tasks), list(peers), own_model
        self.num_peers = len(self.peer_models)
        self.prepare, self.planner_for, self.investigate = prepare, planner_for, investigate
        self.lanes, self.workers, self.log = max(1, int(lanes)), max(1, int(workers)), log
        self.queue = list(range(len(self.tasks)))
        self.lane_task: list = [None] * self.lanes
        self.state: dict[tuple[str, int], TaskState] = {}

    # --- the tasks ------------------------------------------------------------------------------------------------------------
    def _advance(self, tracks: list[Track]) -> None:
        """A lane whose task every track has finished takes the next task, prepared once for all tracks (lanes in parallel)."""
        from feedback_state.swarm import cause_of

        take = []
        for l in range(self.lanes):
            if self.lane_task[l] is not None and not all(self.state[(t.name, l)].done for t in tracks):
                continue
            self.lane_task[l] = self.queue.pop(0) if self.queue else None
            if self.lane_task[l] is not None:
                take.append(l)
        if not take:
            return
        with ThreadPoolExecutor(max_workers=len(take)) as ex:
            logs = list(ex.map(lambda l: self.prepare(l, self.tasks[self.lane_task[l]]), take))
        for l, injection in zip(take, logs):
            task = self.tasks[self.lane_task[l]]
            agent_of = {cause_of(a["profile"]): a for a in task["agents"]}
            order = [c for c in task["labels"] if c in agent_of]
            self.log(f"[marble] lane {l}: task {task['id']} (root causes {task['root_causes']})")
            for t in tracks:
                self.state[(t.name, l)] = TaskState(task=task, planner=self.planner_for(t, l, task), agent_of=agent_of, remaining=list(order),
                                                    injection=injection)

    def events(self, tracks: list[Track]) -> list[tuple]:
        self._advance(tracks)
        active = [(t, l) for t in tracks for l in range(self.lanes)
                  if self.lane_task[l] is not None and not self.state[(t.name, l)].done]
        ask = [(t, l) for t, l in active if not self.state[(t.name, l)].forced]

        def assign(tl):
            try:
                return dict(self.state[(tl[0].name, tl[1])].planner.assign() or {})
            except Exception as e:   # a reply the planner's JSON parser rejects costs the planner its say, not the iteration
                self.log(f"[marble] {tl[0].name}: the planner failed to assign ({type(e).__name__}: {str(e)[:120]})")
                return {}

        with ThreadPoolExecutor(max_workers=max(1, len(ask))) as ex:
            assigned = dict(zip([(t.name, l) for t, l in ask], ex.map(assign, ask)))
        batch = []
        for t, l in active:
            st = self.state[(t.name, l)]
            named = {_key(k): str(v) for k, v in assigned.get((t.name, l), {}).items()}
            by_agent = {_key(a["agent_id"]): c for c, a in st.agent_of.items()}
            causes = [] if st.forced else list(dict.fromkeys(by_agent[k] for k in named if k in by_agent and by_agent[k] in st.remaining))
            if not causes:
                causes = list(st.remaining) if st.forced else st.remaining[:1]
            for cause in causes:
                agent = st.agent_of[cause]
                text = named.get(_key(agent["agent_id"]))
                st.remaining.remove(cause)
                batch.append((t, l, cause_event(st.task, agent, cause, text or default_assignment(cause), st.iteration, text is None)))
        return batch

    # --- the answers ---------------------------------------------------------------------------------------------------------
    def _one(self, item, model: str) -> dict:
        t, l, ev = item
        st = self.state[(t.name, l)]
        try:
            out = self.investigate(l, model, st.task, st.agent_of[ev["cause"]], ev["assignment"])
            return {"response": str(out.get("text") or ""), "evidence": out.get("evidence", "")}
        except Exception as e:   # an investigator that fails has no verdict: its label is 0
            self.log(f"[marble] {model} on {ev['id']}: {type(e).__name__}: {str(e)[:160]}")
            return {"response": f"(the investigation failed: {type(e).__name__})", "evidence": ""}

    def peers(self, items):
        jobs = [(k, p) for k in range(len(items)) for p in range(self.num_peers)]
        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(jobs)))) as ex:
            got = list(ex.map(lambda kp: self._one(items[kp[0]], self.peer_models[kp[1]]), jobs))
        return [got[k * self.num_peers:(k + 1) * self.num_peers] for k in range(len(items))]

    def own(self, items):
        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(items)))) as ex:
            return [a["response"] for a in ex.map(lambda it: self._one(it, self.own_model), items)]

    def display(self, ev: dict, text: str) -> str:
        from pipeline.streams import peer_text

        return peer_text(text)

    def grade(self, ev: dict, text: str) -> int:
        return grade(ev, text)

    # --- the commits -----------------------------------------------------------------------------------------------------------
    def commit(self, t: Track, l: int, ev: dict, text: str, row: dict) -> None:
        st = self.state[(t.name, l)]
        st.verdicts[ev["cause"]] = text
        st.results.append({ev["agent_id"]: text})
        st.tick.append({ev["agent_id"]: text})
        row.update(task_id=ev["task_id"], cause=ev["cause"], iteration=ev["iteration"], forced=ev["forced"], assignment=ev["assignment"][:600],
                   truth=ev["answer"], verdict=verdict(text))
        if "peer_answers" in row:
            row["peer_verdicts"] = [verdict(a) for a in row["peer_answers"]]
        if "own_answer" in row:
            row["own_verdict"] = verdict(row["own_answer"])

    def end_tick(self, tracks: list[Track]) -> None:
        """Every planner that got verdicts this tick reads them; a task every cause of which is done gets its final decision."""
        jobs = [(t, l) for t in tracks for l in range(self.lanes) if (t.name, l) in self.state and self.state[(t.name, l)].tick]
        with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as ex:
            list(ex.map(lambda tl: self._after(*tl), jobs))

    def _after(self, t: Track, l: int) -> None:
        from feedback_state.swarm import summarize_results

        st = self.state[(t.name, l)]
        results, st.tick = st.tick, []
        try:
            st.planner.update(summarize_results(results))
        except Exception as e:
            self.log(f"[marble] {t.name}: the planner failed to update ({type(e).__name__}: {str(e)[:120]})")
        st.iteration += 1
        if st.remaining:
            if not st.forced:
                try:
                    go_on = bool(st.planner.decide(results))
                except Exception:   # the benchmark's engine continues when the decision cannot be read
                    go_on = True
                if not go_on or st.iteration >= int(st.task.get("max_iterations", 5)):
                    st.forced = True
            return
        try:
            final = str(st.planner.final(summarize_results(st.results)) or "")
        except Exception as e:
            self.log(f"[marble] {t.name}: the planner's final decision failed ({type(e).__name__}: {str(e)[:120]})")
            final = ""
        task = st.task
        forced = [r["cause"] for r in t.rows if r.get("task_id") == task["id"] and r.get("forced")]
        t.units.append({"task_id": task["id"], "scenario": task["scenario"], "root_causes": list(task["root_causes"]),
                        "number_of_labels_pred": int(task["number_of_labels_pred"]), "iterations": st.iteration, "forced": forced,
                        "verdicts": {c: verdict(v) for c, v in st.verdicts.items()}, "final": final, "planner": scored(final, task),
                        "verdict_diagnosis": scored(diagnosis(st.verdicts, task["labels"]), task), "injection": st.injection})
        st.done = True

    # --- the results ---------------------------------------------------------------------------------------------------------
    def progress(self, t: Track) -> str:
        return f"{sum(u['planner']['correct'] for u in t.units)}/{len(t.units)} tasks hit by the planner"

    def unit_metrics(self, t: Track) -> dict:
        rows, units = t.rows, t.units
        pos = [r["correct"] for r in rows if r.get("truth") == "yes"]
        neg = [r["correct"] for r in rows if r.get("truth") == "no"]
        out = {"tasks": len(units), "balanced_accuracy": float(np.mean([np.mean(pos), np.mean(neg)])) if pos and neg else None,
               "yes_recall": float(np.mean(pos)) if pos else None, "no_recall": float(np.mean(neg)) if neg else None,
               "forced_share": float(np.mean([bool(r.get("forced")) for r in rows])) if rows else None}
        for key in ("planner", "verdict_diagnosis"):
            if units:
                out[key] = {"accuracy": float(np.mean([u[key]["correct"] for u in units])), "exact": float(np.mean([u[key]["exact"] for u in units])),
                            "set_f1": float(np.mean([u[key]["f1"] for u in units])), "guesses": float(np.mean([len(u[key]["predicted"]) for u in units]))}
        return out


# --- the benchmark's roles on MARBLE (feedback_state.swarm), used by pipeline.marble_online ------------------------------------
class MarblePlanner:
    """The benchmark's EnginePlanner over a task's five agent profiles, the central model behind it (litellm, patch_llm)."""

    def __init__(self, task: dict, llm: str, env):
        from feedback_state.swarm import marble_config, marble_on_path

        marble_on_path()
        from marble.agent.base_agent import BaseAgent
        from marble.engine.engine_planner import EnginePlanner
        from marble.graph.agent_graph import AgentGraph
        from marble.memory.shared_memory import SharedMemory

        config = marble_config(task, llm)
        agents = [BaseAgent(config=a, env=env, model=llm) for a in task["agents"]]   # the profiles it assigns; they never act
        graph = AgentGraph(agents, config)
        for a in agents:
            a.set_agent_graph(graph)
        self.task = task
        self.planner = EnginePlanner(graph, SharedMemory(), config.engine_planner, task["task"], model=llm)

    def assign(self) -> dict:
        return dict(self.planner.assign_tasks(planning_method="naive").get("tasks") or {})

    def update(self, summary: str) -> None:
        self.planner.update_progress(summary)

    def decide(self, results: list[dict]) -> bool:
        return bool(self.planner.decide_next_step(results))

    def final(self, summary: str) -> str:
        return self.planner.summarize_output(summary, self.task["task"], self.task["output_format"]).content or ""


def investigate_marble(env, llm: str, task: dict, agent: dict, assignment: str, queries: int, reasoning: bool = False) -> dict:
    """One investigator of one cause: a MARBLE agent (the cause's profile, its own memory) acting `queries` times on the
    planner's assignment (each act one query_db statement), then its finding (feedback_state.swarm.finding_prompt)."""
    from feedback_state.swarm import cause_of, clipped_memory, finding_prompt, marble_config, marble_on_path

    marble_on_path()
    from marble.agent.base_agent import BaseAgent
    from marble.graph.agent_graph import AgentGraph
    from marble.llms.model_prompting import model_prompting

    cfg = {"agent_id": agent["agent_id"], "profile": agent["profile"]}
    a = BaseAgent(config=cfg, env=env, model=llm)
    a.set_agent_graph(AgentGraph([a], marble_config(task, llm, relationships=[], agents=[cfg])))
    errors = []
    for k in range(int(queries)):
        try:
            a.act(f"{assignment}\nThis is investigation step {k + 1} of {queries}: query the database (one query) for evidence you do not have "
                  f"yet, using your memory of earlier results.")
        except Exception as e:   # the benchmark's engine logs a failed action and moves on
            errors.append(f"step {k + 1}: {type(e).__name__}: {str(e)[:200]}")
    memory = clipped_memory(a)
    prompt = finding_prompt(a.agent_id, a.profile, cause_of(a.profile), memory, answer_line=True)
    text = model_prompting(llm_model=llm, messages=[{"role": "user", "content": prompt}], return_num=1,
                           max_token_num=2048 if reasoning else 400, temperature=0.0)[0].content or ""
    return {"text": text, "evidence": memory, "errors": errors}
