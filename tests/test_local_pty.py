import asyncio
import time

from tamfis_code.pty import LocalPtyBroker


def test_local_pty_drains_output_and_supports_input(tmp_path):
    async def run():
        broker = LocalPtyBroker(cwd=str(tmp_path))
        session = broker.start("read line; printf 'reply:%s\\n' \"$line\"")
        try:
            broker.write(session.id, "hello\n")
            deadline = time.monotonic() + 2
            data = ""
            while time.monotonic() < deadline and "reply:hello" not in data:
                await asyncio.sleep(0.02)
                _, data, _ = broker.read(session.id)
            assert "reply:hello" in data
        finally:
            broker.close()

    asyncio.run(run())


def test_local_pty_output_is_bounded(tmp_path):
    async def run():
        broker = LocalPtyBroker(cwd=str(tmp_path))
        broker.MAX_BUFFER = 128
        session = broker.start("python -c 'print(\"x\" * 1000)'")
        try:
            await asyncio.sleep(0.1)
            assert len(session.output) <= 128
        finally:
            broker.close()

    asyncio.run(run())
