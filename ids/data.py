"""Loading the NSL-KDD dataset and mapping raw attack names to categories."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}

# The 41 connection features defined by the KDD Cup '99 / NSL-KDD dataset.
FEATURE_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
ALL_COLUMNS = FEATURE_COLUMNS + ["attack", "difficulty"]

# Alert categories shown to an analyst. NSL-KDD's R2L and U2R families are
# both "someone trying to get access they shouldn't have" (password guessing,
# FTP/IMAP exploits, privilege escalation), so they're grouped as Brute Force.
CATEGORIES = ["Normal", "DoS", "Probe", "Brute Force"]

ATTACK_CATEGORY = {
    "normal": "Normal",
    # Denial of service: flood or crash a host so it can't serve real users.
    **dict.fromkeys(
        ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
         "mailbomb", "processtable", "udpstorm"], "DoS"),
    # Probe: scanning to map hosts, ports and services before an attack.
    **dict.fromkeys(
        ["ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"], "Probe"),
    # Remote-to-local and user-to-root: guessing or exploiting into access.
    **dict.fromkeys(
        ["ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy",
         "warezclient", "warezmaster", "named", "sendmail", "snmpgetattack",
         "snmpguess", "worm", "xlock", "xsnoop", "httptunnel",
         "buffer_overflow", "loadmodule", "perl", "rootkit", "ps", "sqlattack",
         "xterm"], "Brute Force"),
}


def download(data_dir="data"):
    """Download the NSL-KDD train and test files if they aren't there yet."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = data_dir / name
        if not target.exists():
            print(f"Downloading {name} ...")
            urlretrieve(BASE_URL + name.replace("+", "%2B"), target)
    return data_dir


def load(split, data_dir="data"):
    """Read one split ("train" or "test") as a DataFrame with named columns."""
    path = Path(data_dir) / FILES[split]
    return pd.read_csv(path, header=None, names=ALL_COLUMNS)


def to_category(attack_names):
    """Map raw attack names (e.g. "neptune") to an alert category (e.g. "DoS")."""
    categories = attack_names.map(ATTACK_CATEGORY)
    unknown = sorted(attack_names[categories.isna()].unique())
    if unknown:
        raise ValueError(f"Unmapped attack names: {unknown}")
    return categories
