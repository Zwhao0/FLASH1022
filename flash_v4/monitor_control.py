"""Opt-in one-worker Gloo control group; collective semantics stay unchanged."""
from datetime import timedelta
import os

CONTRACT = 'r92_gloo1_v1'
BACKEND = 'flash_gloo1'


def validate_contract(value):
    if value not in (CONTRACT, 'r92_store_v1'):
        raise ValueError('unsupported MONITOR control contract')
    return value


def create_group(dist, contract):
    if validate_contract(contract) != CONTRACT:
        raise ValueError('Gloo control requires its own contract')
    # This runtime exposes Gloo's options but new_group(backend='gloo') ignores
    # pg_options. A registered backend keeps normal group membership, timeout,
    # broadcast rendezvous, error handling and destroy_process_group lifecycle.
    if not hasattr(dist.Backend, BACKEND.upper()):
        def create(store, rank, world_size, timeout):
            options = dist.ProcessGroupGloo._Options()
            options._timeout = timeout
            options._threads = 1
            interfaces = os.environ.get('GLOO_SOCKET_IFNAME', '').split(',')
            if not interfaces or any(not item.strip() for item in interfaces):
                raise ValueError('explicit Gloo interface required')
            options._devices = [dist.ProcessGroupGloo.create_device(interface=item.strip())
                                for item in interfaces]
            return dist.ProcessGroupGloo(store, rank, world_size, options)
        dist.Backend.register_backend(BACKEND, create, devices=['cpu'])
    return dist.new_group(backend=BACKEND, timeout=timedelta(seconds=30))
