"""
Posterior Storage for Sequential Bayesian Updating

Manages saving and loading of posterior samples for each event.
Enables sequential updating: load previous posterior → use as prior for next fit.

Storage Format:
- NetCDF files via arviz: {event_id}_{timestamp}.nc
- Keep only latest N posteriors per event (auto-cleanup)
- Directory: models/bayesian_gamma_model/posteriors/
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List
import arviz as az
import pandas as pd


class PosteriorStore:
    """
    Manage storage of posterior samples for sequential updating.
    """

    def __init__(
        self,
        base_dir: str = "models/bayesian_gamma_model/posteriors",
        max_posteriors_per_event: int = 5
    ):
        """
        Initialize posterior store.

        Args:
            base_dir: Directory to store posteriors
            max_posteriors_per_event: Keep only N most recent per event
        """
        self.base_dir = Path(base_dir)
        self.max_posteriors_per_event = max_posteriors_per_event

        # Create directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_posterior(
        self,
        event_id: str,
        fit_date: pd.Timestamp,
        idata: az.InferenceData
    ) -> str:
        """
        Save posterior samples for an event.

        Args:
            event_id: Event identifier
            fit_date: Date of fit
            idata: InferenceData with posterior samples

        Returns:
            Path to saved file
        """
        # Sanitize event_id for filename
        safe_event_id = self._sanitize_filename(event_id)

        # Create filename with timestamp
        timestamp = fit_date.strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_event_id}_{timestamp}.nc"
        filepath = self.base_dir / filename

        # Save as NetCDF
        try:
            az.to_netcdf(idata, filepath)
        except Exception as e:
            print(f"Failed to save posterior for {event_id}: {e}")
            raise

        # Cleanup old posteriors
        self._cleanup_old_posteriors(safe_event_id)

        return str(filepath)

    def load_latest_posterior(
        self,
        event_id: str
    ) -> Optional[Tuple[pd.Timestamp, az.InferenceData]]:
        """
        Load most recent posterior for an event.

        Args:
            event_id: Event identifier

        Returns:
            (fit_date, idata) or None if no posterior exists
        """
        safe_event_id = self._sanitize_filename(event_id)

        # Find all files for this event
        pattern = f"{safe_event_id}_*.nc"
        files = sorted(self.base_dir.glob(pattern), reverse=True)

        if not files:
            return None

        # Load most recent
        latest_file = files[0]

        try:
            idata = az.from_netcdf(latest_file)
        except Exception as e:
            print(f"Failed to load posterior from {latest_file}: {e}")
            # Try to delete corrupted file
            try:
                latest_file.unlink()
            except:
                pass
            return None

        # Extract timestamp from filename
        timestamp_str = latest_file.stem.split('_')[-2:]  # ['YYYYMMDD', 'HHMMSS']
        timestamp_str = '_'.join(timestamp_str)

        try:
            fit_date = pd.Timestamp(datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S"))
        except:
            # Fallback to file modification time
            fit_date = pd.Timestamp(datetime.fromtimestamp(latest_file.stat().st_mtime))

        return fit_date, idata

    def has_prior(self, event_id: str) -> bool:
        """
        Check if a prior posterior exists for this event.

        Args:
            event_id: Event identifier

        Returns:
            True if posterior exists
        """
        return self.load_latest_posterior(event_id) is not None

    def delete_event_posteriors(self, event_id: str):
        """
        Delete all posteriors for an event.

        Args:
            event_id: Event identifier
        """
        safe_event_id = self._sanitize_filename(event_id)
        pattern = f"{safe_event_id}_*.nc"

        for filepath in self.base_dir.glob(pattern):
            try:
                filepath.unlink()
            except Exception as e:
                print(f"Failed to delete {filepath}: {e}")

    def _cleanup_old_posteriors(self, safe_event_id: str):
        """
        Keep only N most recent posteriors for an event.

        Args:
            safe_event_id: Sanitized event identifier
        """
        pattern = f"{safe_event_id}_*.nc"
        files = sorted(self.base_dir.glob(pattern), reverse=True)

        # Delete older files beyond max limit
        for old_file in files[self.max_posteriors_per_event:]:
            try:
                old_file.unlink()
            except Exception as e:
                print(f"Failed to delete old posterior {old_file}: {e}")

    def _sanitize_filename(self, event_id: str) -> str:
        """
        Sanitize event ID for use in filename.

        Replaces special characters with underscores.
        """
        # Replace special chars
        safe = event_id.replace('/', '_').replace('\\', '_')
        safe = safe.replace(':', '_').replace(' ', '_')
        safe = safe.replace('<', '_').replace('>', '_')
        safe = safe.replace('|', '_').replace('?', '_')
        safe = safe.replace('*', '_').replace('"', '_')

        # Truncate if too long
        max_len = 100
        if len(safe) > max_len:
            safe = safe[:max_len]

        return safe

    def list_events(self) -> List[str]:
        """
        List all events with stored posteriors.

        Returns:
            List of event IDs (sanitized)
        """
        files = self.base_dir.glob("*.nc")
        event_ids = set()

        for f in files:
            # Extract event_id from filename (before last underscore group)
            parts = f.stem.split('_')
            # Last two parts are timestamp (YYYYMMDD_HHMMSS)
            event_id = '_'.join(parts[:-2])
            event_ids.add(event_id)

        return sorted(list(event_ids))

    def get_posterior_count(self, event_id: str) -> int:
        """
        Count stored posteriors for an event.

        Args:
            event_id: Event identifier

        Returns:
            Number of stored posteriors
        """
        safe_event_id = self._sanitize_filename(event_id)
        pattern = f"{safe_event_id}_*.nc"
        return len(list(self.base_dir.glob(pattern)))
