from ..stateful import collector as base
from ..stateful.predictor import MockPredictor
from .environment import HardenedEnvironment,verify
from .tasks import VERSION,Oracle
from .protocol import prompt_for,schema_hash,STOP_PROTOCOLS

class OracleMock(MockPredictor):
    # Keep mock provenance explicit; replace only reference-plan resolution for new tasks.
    def __init__(self,task,responses=None):
        super().__init__(task,responses)
        self.oracle=Oracle(task)

def replay(task,trace,state_observer=None):
    # Reuse the original recorded-action Replay with extension dependencies, no network calls.
    return base.replay(task,trace,environment_factory=HardenedEnvironment,verifier=verify,state_observer=state_observer)

def run_episode(task,predictor,clean=False,max_tool_calls=12,max_requests=16,
                stop_protocol="legacy",stop_on_verified_success=False):
    from .analysis import annotate
    if stop_protocol not in STOP_PROTOCOLS:
        raise ValueError("Unknown stop protocol")
    builder=lambda task,events,remaining,feedback:prompt_for(task,events,remaining,feedback,stop_protocol)
    trace=base.run_episode(task,predictor,clean=clean,max_tool_calls=max_tool_calls,max_requests=max_requests,
        environment_factory=HardenedEnvironment,verifier=verify,prompt_builder=builder,replay_function=replay,stop_on_verified_success=stop_on_verified_success)
    trace.update(dataset_version=VERSION,eval_group=task["eval_group"],variant_id=task["variant_id"],
                 schema_hash=schema_hash(),collection_limits={"max_tool_calls":max_tool_calls,"max_requests":max_requests})
    if stop_protocol!="legacy" or stop_on_verified_success:
        trace["termination_protocol"]={"prompt_version":stop_protocol,"stop_on_verified_success":bool(stop_on_verified_success)}
    annotate(task,trace)
    return trace
