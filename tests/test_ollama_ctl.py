import unittest
from unittest.mock import Mock, patch


class OllamaCtlTests(unittest.TestCase):
    def test_restart_stops_starts_waits_and_cools_down(self):
        import ollama_ctl

        proc = Mock()
        with patch.object(ollama_ctl, "stop") as stop, patch.object(ollama_ctl.subprocess, "Popen", return_value=proc) as popen, patch.object(ollama_ctl, "wait") as wait, patch.object(ollama_ctl.time, "sleep") as sleep:
            result = ollama_ctl.restart(5)

        stop.assert_called_once_with()
        popen.assert_called_once_with(["ollama", "serve"], stdout=ollama_ctl.subprocess.DEVNULL, stderr=ollama_ctl.subprocess.DEVNULL)
        wait.assert_called_once_with()
        sleep.assert_called_once_with(5)
        self.assertIs(result, proc)


if __name__ == "__main__":
    unittest.main()
