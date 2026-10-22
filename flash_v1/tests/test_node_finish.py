"""Regression for the remote drain versus private Redis shutdown race."""
import threading
import unittest

from flash_v1.scripts.node_collection import online_finish_barrier


class NodeFinishTests(unittest.TestCase):
    def test_controller_and_both_drains_finish_before_server_shutdown(self):
        class Redis:
            def __init__(self):
                self.values, self.lock = {}, threading.Lock()
                self.closed = False
                self.peer_acknowledged = False
                self.peer_waiting = threading.Event()

            def set(self, key, value):
                with self.lock:
                    if self.closed:
                        raise AssertionError('publication after Redis shutdown')
                    if threading.current_thread().name == 'node1' and self.peer_acknowledged:
                        raise AssertionError('node1 request after final ACK')
                    self.values[key] = value
                    if key.endswith('drain_node1'):
                        self.peer_acknowledged = True

            def exists(self, key):
                with self.lock:
                    if self.closed:
                        raise AssertionError('request after Redis shutdown')
                    if threading.current_thread().name == 'node1':
                        if self.peer_acknowledged:
                            raise AssertionError('node1 request after final ACK')
                        self.peer_waiting.set()
                    return key in self.values

        client = Redis()
        peer_drain_started = threading.Event()
        allow_peer_drain_finish = threading.Event()
        server_closed = threading.Event()
        errors = []

        def peer():
            try:
                online_finish_barrier(client, 'session', 1, timeout=3)
                peer_drain_started.set()
                if not allow_peer_drain_finish.wait(3):
                    raise AssertionError('test did not release peer drain')
                online_finish_barrier(client, 'session', 1, local_drained=True, timeout=3)
            except BaseException as error:
                errors.append(error)

        def owner():
            try:
                online_finish_barrier(client, 'session', 0, local_drained=True, timeout=3)
                with client.lock:
                    self.assertTrue(client.values.get('online_finish:session:drain_node1'))
                    client.closed = True
                server_closed.set()
            except BaseException as error:
                errors.append(error)

        peer_thread = threading.Thread(target=peer, name='node1')
        owner_thread = threading.Thread(target=owner, name='node0')
        peer_thread.start()
        try:
            self.assertTrue(client.peer_waiting.wait(2))
            self.assertFalse(peer_drain_started.is_set(), 'peer stopped draining before controller completion')
            online_finish_barrier(client, 'session', 0, timeout=3)
            self.assertTrue(peer_drain_started.wait(2))
            owner_thread.start()
            self.assertFalse(server_closed.wait(.05), 'owner closed Redis before the peer final drain ACK')
        finally:
            allow_peer_drain_finish.set()
            peer_thread.join(4)
            if owner_thread.ident is not None:
                owner_thread.join(4)
        self.assertFalse(peer_thread.is_alive())
        self.assertFalse(owner_thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(server_closed.is_set())


if __name__ == '__main__':
    unittest.main()
