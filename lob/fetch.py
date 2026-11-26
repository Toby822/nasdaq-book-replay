"""Fetch the three official ten-level LOBSTER samples and verify SHA-256."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

BASE = "https://php.lobsterdata.com/info/sample/LOBSTER_SampleFile_"
CHECKSUMS = {
    "AAPL": "326839316d67d7819ca0541ffdf73cdb044d6ae4105cce0ef9c615e8848e9d43",
    "AMZN": "5cff62a609b27aef82285382ad646c4c2facc0a692a60bedda633de64c4aa54f",
    "MSFT": "0825e00ec83cb8ac8b53fd1efb7f2138b7848659e625c49621ebd9b55757be4c",
}


def main():
    root = Path("data")
    root.mkdir(exist_ok=True)
    for ticker, digest in CHECKSUMS.items():
        file = root / (ticker + ".zip")
        if file.exists():
            contents = file.read_bytes()
        else:
            with urlopen(BASE + ticker + "_2012-06-21_10.zip", timeout=30) as response:
                contents = response.read()
        if hashlib.sha256(contents).hexdigest() != digest:
            raise ValueError(f"Unexpected checksum for {ticker}")
        if not file.exists():
            file.write_bytes(contents)


if __name__ == "__main__":
    main()
