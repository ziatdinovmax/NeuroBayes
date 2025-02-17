import sys
from unittest.mock import MagicMock

# Mock sklearn before any imports happen
mock_sklearn = MagicMock()
mock_kmeans = MagicMock()
mock_sklearn.cluster.KMeans = mock_kmeans
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.cluster'] = MagicMock()