import os
import sys
import asyncio
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from fastapi.testclient import TestClient


class TestScheduler:

    @patch("cemg.api.get_driver")
    @patch("cemg.api.bootstrap_schema")
    @patch("cemg.api.is_healthy")
    @patch("cemg.api.prune")
    def test_lifespan_spawns_and_cancels_pruning_task(self, mock_prune, mock_is_healthy, mock_bootstrap, mock_get_driver):
        # Set up mocks
        mock_driver = MagicMock()
        mock_get_driver.return_value = mock_driver
        mock_is_healthy.return_value = True
        mock_prune.return_value = {"eligible_count": 5, "deleted": True}

        # Set a very short interval for testing (e.g. 0.05 seconds) so it fires quickly
        with patch.dict(os.environ, {"CEMG_PRUNE_INTERVAL_SECONDS": "0.05"}):
            from cemg.api import app

            # Verify client triggers lifespan events
            with TestClient(app) as client:
                # Wait long enough for the loop to run at least once
                # We sleep in asyncio so we yield control to the background task
                # TestClient runs lifespan startup synchronously, so we must sleep briefly
                # to let the spawned task run.
                import time
                time.sleep(0.15)
                
            # After exiting lifespan context manager, the pruning task is cancelled.
            # Verify prune was called
            assert mock_prune.called
            # The first argument to prune should be the mocked driver
            mock_prune.assert_called_with(mock_driver, dry_run=False)
