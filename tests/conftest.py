import sys
from unittest.mock import MagicMock

def mock_modules():
    # Mock duckdb
    duckdb = MagicMock()
    sys.modules["duckdb"] = duckdb

    # Mock numpy
    numpy = MagicMock()
    numpy.__version__ = "1.26.4"
    # Some common numpy functions/constants used
    numpy.mean = MagicMock(return_value=0.0)
    numpy.abs = MagicMock(return_value=numpy)
    numpy.log = MagicMock(return_value=0.0)
    numpy.array = MagicMock(side_effect=lambda x, **kwargs: x)
    sys.modules["numpy"] = numpy

    # Mock pandas
    pandas = MagicMock()
    pandas.__version__ = "2.2.2"
    pandas.to_datetime = MagicMock(side_effect=lambda x, **kwargs: x)
    pandas.Timestamp = MagicMock
    pandas.Timedelta = MagicMock
    pandas.DataFrame = MagicMock
    sys.modules["pandas"] = pandas

    # Mock dotenv
    dotenv = MagicMock()
    sys.modules["dotenv"] = dotenv

    # Mock coinbase
    coinbase = MagicMock()
    sys.modules["coinbase"] = coinbase
    sys.modules["coinbase.rest"] = MagicMock()
    sys.modules["coinbase.websocket"] = MagicMock()

    # Mock torch
    torch = MagicMock()
    torch.__version__ = "2.3.0"
    torch.device = MagicMock
    torch.nn = MagicMock()
    torch.nn.Module = MagicMock
    torch.nn.Linear = MagicMock
    torch.nn.ModuleList = MagicMock
    torch.optim = MagicMock()
    torch.optim.Adam = MagicMock
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.optim"] = torch.optim

mock_modules()
