import json, os, re
with open("tile_compile.schema.json") as f:
    s = json.load(f)
def get_leaf_keys(d, p=""):
    k = []
    if isinstance(d, dict):
        if "properties" in d:
            for k2, v in d["properties"].items():
                k.extend(get_leaf_keys(v, k2))
        else:
            k.append(p)
    return set(k)

keys = [k for k in get_leaf_keys(s) if k]
contents = []
for r, d, f in os.walk("."):
    for n in f:
        if n.endswith(".cpp") or n.endswith(".hpp"):
            p = os.path.join(r, n)
            if "io/config.cpp" in p or "config/configuration.hpp" in p or "cli_main.cpp" in p: continue
            try: contents.append(open(p).read())
            except: pass

unused = [k for k in keys if not any(re.search(r"\b"+re.escape(k)+r"\b", c) for c in contents)]
print("Total keys:", len(keys))
if unused:
    for u in sorted(unused): print("UNUSED:", u)
else:
    print("ALL USED")
