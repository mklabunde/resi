import subprocess


def get_workers_for_current_node() -> int:
    n_workers: int
    hostname = subprocess.getoutput(["hostname"])  # type: ignore

    if hostname in ["hdf19-gpu16", "hdf19-gpu17", "hdf19-gpu18", "hdf19-gpu19"]:
        return 16
    if hostname.startswith("hdf19-gpu") or hostname.startswith("e071-gpu"):
        return 12
    elif hostname.startswith("e230-dgx1"):
        return 10
    elif hostname.startswith("hdf18-gpu"):
        return 16
    elif hostname.startswith("e230-dgx2"):
        return 6
    elif hostname.startswith(("e230-dgxa100-", "lsf22-gpu", "e132-comp")):
        return 32
    elif hostname.startswith("e230-pc14"):  # Own workstation
        return 12  # 24
    else:
        raise NotImplementedError()
