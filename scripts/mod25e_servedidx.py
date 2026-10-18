import inspect
import os

OLD_SHIFT = '            st["shift"] = shift\n'
NEW_SHIFT = OLD_SHIFT + '            st["served"] = int(j)\n'
OLD_APPEND = "            super().append(v if sh is None else (v[0], v[1], sh))"
NEW_APPEND = (
    '            sv = st.pop("served", None)\n'
    "            super().append(v if sh is None else (v[0], v[1] if sv is None else sv, sh))"
)


def enabled():
    return os.environ.get("SERVEDIDX") == "1"


def patched_source():
    import mod25e_gz as gz

    src = inspect.getsource(gz.install_gz)
    assert src.count(OLD_SHIFT) == 1 and src.count(OLD_APPEND) == 1
    return src.replace(OLD_SHIFT, NEW_SHIFT).replace(OLD_APPEND, NEW_APPEND)


def install_gz_served():
    import mod25e_gz as gz

    ns = dict(gz.__dict__)
    exec(compile(patched_source(), gz.__file__, "exec"), ns)
    return ns["install_gz"]()
