from ..stateful import collector as base
from ..stateful.predictor import MockPredictor
from .environment import HardenedEnvironment,verify
from .tasks import VERSION,Oracle
from .protocol import prompt_for,schema_hash

class OracleMock(MockPredictor):
    # Keep mock provenance explicit; replace only reference-plan resolution for new tasks.
    def __init__(self,task,responses=None):
        super().__init__(task,responses)
        self.oracle=Oracle(task)

def replay(task,trace):
    # Reuse the original recorded-action Replay with extension dependencies, no network calls.
    return base.replay(task,trace,environment_factory=HardenedEnvironment,verifier=verify)

def run_episode(task,predictor,clean=False,max_tool_calls=12,max_requests=16):
    from .analysis import annotate
    trace=base.run_episode(task,predictor,clean=clean,max_tool_calls=max_tool_calls,max_requests=max_requests,
        environment_factory=HardenedEnvironment,verifier=verify,prompt_builder=prompt_for,replay_function=replay)
    trace.update(dataset_version=VERSION,eval_group=task["eval_group"],variant_id=task["variant_id"],
                 schema_hash=schema_hash(),collection_limits={"max_tool_calls":max_tool_calls,"max_requests":max_requests})
    annotate(task,trace)
    return trace
