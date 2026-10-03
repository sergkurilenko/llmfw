"""Синтетические мини-репозитории для smoke-теста (DESIGN.md §11) и pytest.

Создаёт в каталоге ``<work>/raw/<split>/<owner>__<name>/`` пять «репозиториев» — protected, hard_neg,
public_train, public_calib, public_test — с файлами на python, c, go и javascript (10–20 функций в каждом).
Функции генерируются из небольшой грамматики операторов с детерминированным ГПСЧ: структура тела
(последовательность и вложенность операторов) случайна, поэтому абстрактные токен-потоки разных
репозиториев почти не пересекаются, а внутри protected есть и короткие, и длинные функции.
hard_neg использует словарь той же «предметной области» (cipher/tunnel/peer), что и protected.

Также пишет smoke-конфиг (``smoke.yaml``): все пути внутри ``<work>``, маленькие k/w отпечатков,
небольшая выборка функций, короткий бутстрэп — конвейер 01→02→03→04→07→09 проходит на CPU за секунды.

CLI (из корня стенда):
    python tests/synthetic_repos.py --out <work> [--seed N]   # печатает путь к smoke.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск как скрипта: python tests/synthetic_repos.py
    sys.path.insert(0, str(ROOT))

SMOKE_SEED = 20261003
SMOKE_CONFIG_NAME = "smoke.yaml"
LANGS: tuple[str, ...] = ("python", "c", "go", "javascript")

# split -> (каталог клона owner__name, словарь имён)
REPOS: dict[str, tuple[str, str]] = {
    "protected": ("acme__vpncore", "secure"),
    "hard_neg": ("other__netguard", "secure"),
    "public_train": ("foo__webkit", "web"),
    "public_calib": ("bar__gametools", "game"),
    "public_test": ("baz__dataforge", "data"),
}
# относительные пути файлов (без доменных токенов в public_*, см. extract.DOMAIN_PATH_*)
FILE_PATHS: dict[str, str] = {
    "python": "src/{stem}.py",
    "c": "src/{stem}.c",
    "go": "pkg/{stem}.go",
    "javascript": "lib/{stem}.js",
}
VOCAB: dict[str, dict[str, list[str]]] = {
    "secure": {
        "nouns": ["cipher", "tunnel", "packet", "nonce", "digest", "session", "peer", "handshake", "frame", "route", "policy", "ticket"],
        "verbs": ["seal", "open", "rotate", "verify", "derive", "wrap", "unwrap", "inspect", "flush", "negotiate", "pad", "scan"],
        "strings": ["bad nonce", "tunnel closed", "peer timeout", "digest mismatch", "handshake", "replay", "policy", "ticket"],
    },
    "web": {
        "nouns": ["request", "route", "template", "header", "cookie", "view", "form", "layout", "widget", "page", "asset", "menu"],
        "verbs": ["render", "dispatch", "collect", "format", "paginate", "escape", "bind", "resolve", "merge", "trim", "count", "sort"],
        "strings": ["text/html", "not found", "layout", "index", "menu", "widget", "form", "page"],
    },
    "game": {
        "nouns": ["sprite", "tile", "score", "player", "level", "enemy", "bonus", "camera", "sound", "quest", "inventory", "map"],
        "verbs": ["spawn", "move", "collide", "award", "advance", "respawn", "animate", "shake", "mute", "complete", "equip", "load"],
        "strings": ["game over", "level up", "bonus", "spawn", "quest done", "camera", "player", "tile"],
    },
    "data": {
        "nouns": ["frame", "column", "row", "batch", "schema", "bucket", "cursor", "record", "shard", "chunk", "metric", "sample"],
        "verbs": ["aggregate", "filter", "project", "shuffle", "partition", "validate", "compact", "encode", "bucketize", "sample", "join", "rank"],
        "strings": ["schema", "null column", "batch", "shard", "metric", "cursor", "record", "chunk"],
    },
}
PARAM_NAMES = ["a", "b", "n", "limit", "offset", "count", "size", "flag", "mode", "width"]
LOCAL_NAMES = ["acc", "total", "tmp", "cur", "prev", "idx", "step", "mask", "left", "right", "seen", "best", "delta", "val", "res", "span"]


class _Gen:
    """Генератор функций одного языка: тело — случайная последовательность операторов грамматики."""

    def __init__(self, lang: str, rng: random.Random, vocab: dict[str, list[str]]):
        self.lang = lang
        self.rng = rng
        self.vocab = vocab
        self.locals: list[str] = []
        self.params: list[str] = []
        self.has_items = False
        self.has_buf = False

    # ------------------------------------------------------------------ helpers
    def num(self, lo: int = 1, hi: int = 97) -> str:
        return str(self.rng.randint(lo, hi))

    def hexnum(self) -> str:
        return "0x%02x" % self.rng.randint(1, 255)

    def string(self) -> str:
        return self.rng.choice(self.vocab["strings"]) + " " + self.rng.choice(["ok", "fail", "ready", "busy", "x", "y"])

    def var(self) -> str:
        pool = self.locals + self.params
        return self.rng.choice(pool) if pool else self.params[0]

    def new_local(self) -> str:
        cands = [n for n in LOCAL_NAMES if n not in self.locals and n not in self.params]
        name = self.rng.choice(cands) if cands else f"v{len(self.locals)}"
        self.locals.append(name)
        return name

    def operand(self, depth: int = 0) -> str:
        """Операнд выражения: переменная, литерал, вызов/приведение, индекс или скобочное подвыражение."""
        r = self.rng.random()
        if depth < 2 and r < 0.18:
            return f"({self.expr(depth + 1)})"
        if r < 0.45:
            return self.var()
        if r < 0.62:
            return self.num()
        if r < 0.74:
            return self.cast(self.var())
        if r < 0.86:
            return self.index_expr()
        return self.unary(self.var())

    def cast(self, v: str) -> str:
        return {
            "python": self.rng.choice([f"int({v})", f"abs({v})", f"max({v}, {self.num()})", f"min({v}, {self.num()})", f"({v} or {self.num()})"]),
            "c": self.rng.choice([f"(unsigned){v}", f"(long){v}", f"abs({v})", f"(int)({v} & 0xff)", f"({v} ? {v} : {self.num()})"]),
            "go": self.rng.choice([f"int64({v})", f"uint32({v})", f"int({v} & 0xff)", f"int(byte({v}))", f"({v} * {self.num()})"]),
            "javascript": self.rng.choice([f"Number({v})", f"Math.abs({v})", f"Math.max({v}, {self.num()})", f"({v} | 0)", f"Math.floor({v} / {self.num(2, 9)})"]),
        }[self.lang]

    def index_expr(self) -> str:
        if self.lang == "python":
            return self.rng.choice([f"{self.var()} % {self.num(2, 31)}", f"len(str({self.var()}))", f"[{self.var()}, {self.num()}][{self.num(0, 1)}]"])
        if self.lang == "c":
            return self.rng.choice([f"{self.var()} % {self.num(2, 31)}", f"({self.var()} >> {self.num(1, 7)})", f"sizeof(int) * {self.var()}"])
        if self.lang == "go":
            return self.rng.choice([f"{self.var()} % {self.num(2, 31)}", f"({self.var()} >> {self.num(1, 7)})", f"len([]int{{{self.var()}, {self.num()}}})"])
        return self.rng.choice([f"{self.var()} % {self.num(2, 31)}", f"({self.var()} >> {self.num(1, 7)})", f"[{self.var()}, {self.num()}].length"])

    def unary(self, v: str) -> str:
        if self.lang == "python":
            return self.rng.choice([f"-{v}", f"~{v}", f"(not {v})", f"+{v}"])
        if self.lang == "go":
            return self.rng.choice([f"-{v}", f"^{v}", f"+{v}"])
        return self.rng.choice([f"-{v}", f"~{v}", f"!{v}", f"+{v}"])

    def expr(self, depth: int = 0) -> str:
        """Случайное дерево выражения из 1–4 операндов (разные формы абстрактных токенов)."""
        n_ops = self.rng.choice([1, 2, 2, 3, 3, 4])
        ops_pool = ["+", "-", "*", "|", "&", "^"]
        if self.lang != "python":
            ops_pool += ["<<", ">>"]
        else:
            ops_pool += ["//", "%", "<<"]
        parts = [self.operand(depth)]
        for _ in range(n_ops - 1):
            op = self.rng.choice(ops_pool)
            rhs = self.operand(depth)
            if op in ("<<", ">>", "//", "%"):
                rhs = self.num(1, 7) if op in ("<<", ">>") else self.num(2, 31)
            parts.append(f"{op} {rhs}")
        return " ".join(parts)

    def cond(self) -> str:
        """Условие: простое сравнение, составное (and/or), остаток, отрицание."""
        cmp = self.rng.choice([">", "<", ">=", "<=", "!=", "=="])
        land, lor, lnot = {"python": ("and", "or", "not "), "c": ("&&", "||", "!"), "go": ("&&", "||", "!"), "javascript": ("&&", "||", "!")}[self.lang]
        r = self.rng.random()
        if r < 0.35:
            return f"{self.var()} {cmp} {self.num()}"
        if r < 0.55:
            return f"{self.var()} {cmp} {self.var()}"
        if r < 0.75:
            return f"{self.var()} {cmp} {self.num()} {self.rng.choice([land, lor])} {self.var()} {self.rng.choice(['<', '>', '!='])} {self.num()}"
        if r < 0.9:
            return f"{self.var()} % {self.num(2, 13)} == {self.num(0, 3)}"
        return f"{lnot}({self.var()} {cmp} {self.num()})"

    # ------------------------------------------------------------------ public
    def function(self, name: str, n_stmts: int) -> str:
        self.locals, self.params, self.has_items, self.has_buf = [], [], False, False
        n_params = self.rng.randint(1, 3)
        self.params = self.rng.sample(PARAM_NAMES, n_params)
        emit: dict[str, Callable[[str, int], list[str]]] = {
            "python": self._py, "c": self._c, "go": self._go, "javascript": self._js,
        }
        return "\n".join(emit[self.lang](name, n_stmts)) + "\n"

    def _stmts(self, n: int, depth: int, kinds: dict[str, Callable[[int], list[str]]]) -> list[str]:
        """n операторов; во вложенных блоках нет составных операторов и есть хотя бы один не-комментарий."""
        out: list[str] = []
        names = list(kinds)
        real = 0
        for _ in range(n):
            k = self.rng.choice(names)
            if depth > 0 and k in ("if", "for", "while", "try", "switch"):
                k = self.rng.choice(["assign", "update", "call", "ternary"])
            if k != "comment":
                real += 1
            out.extend(kinds[k](depth))
        if depth > 0 and real == 0:
            out.extend(kinds["update"](depth))
        return out

    # ------------------------------------------------------------------ python
    def _py(self, name: str, n: int) -> list[str]:
        ind = "    "

        def block(depth: int) -> list[str]:
            return [ind + s for s in self._stmts(self.rng.randint(1, 3), depth + 1, kinds)]

        def s_assign(d: int) -> list[str]:
            v = self.new_local()
            return [f"{v} = {self.expr()}"]

        def s_update(d: int) -> list[str]:
            if not self.locals:
                return s_assign(d)
            v = self.rng.choice(self.locals)
            return [f"{v} {self.rng.choice(['+=', '-=', '*=', '^=', '|=', '&=', '//=', '%='])} {self.operand()}"]

        def s_if(d: int) -> list[str]:
            v = self.new_local()
            lines = [f"if {self.cond()}:"] + block(d)
            r = self.rng.random()
            if r < 0.3:
                lines += [f"elif {self.cond()}:", f"{ind}{v} = {self.expr()}"]
            if r < 0.7:
                lines += ["else:", f"{ind}{v} = {self.operand()}"]
            return lines

        def s_for(d: int) -> list[str]:
            if self.has_items and self.rng.random() < 0.4:
                tgt = self.rng.choice(self.locals or self.params)
                return [self.rng.choice(["for item in items:", "for pos, item in enumerate(items):", "for item in reversed(items):"]),
                        f"{ind}{tgt} {self.rng.choice(['=', '+=', '^='])} item {self.rng.choice(['+', '*', '-', '|'])} {self.operand()}"] + block(d)
            rng_expr = self.rng.choice([f"range({self.num(2, 40)})", f"range({self.operand()}, {self.num(40, 90)}, {self.num(1, 5)})",
                                        f"range(len(str({self.var()})))", f"range({self.var()} % {self.num(2, 9)}, {self.num(10, 30)})"])
            return [f"for i in {rng_expr}:"] + block(d)

        def s_while(d: int) -> list[str]:
            v = self.new_local()
            init = self.rng.choice([self.num(0, 5), self.operand(), f"{self.var()} % {self.num(2, 7)}"])
            step = self.rng.choice([f"{v} += {self.num(1, 9)}", f"{v} = {v} * 2 + {self.num()}", f"{v} += {self.operand()} + 1"])
            return [f"{v} = {init}", f"while {v} < {self.num(10, 60)}{self.rng.choice(['', ' and ' + self.cond()])}:", f"{ind}{step}"] + block(d)

        def s_call(d: int) -> list[str]:
            v = self.new_local()
            fn = self.rng.choice(["len", "max", "min", "abs", "sum", "str", "hash", "sorted", "divmod", "pow", "round"])
            if fn in ("len", "sum", "sorted"):
                add = self.rng.choice([f"items.append({self.expr()})", f"items.extend([{self.operand()}, {self.operand()}])",
                                       f"items.insert(0, {self.operand()})", f"items += [{self.operand()}]"])
                use = self.rng.choice([f"{fn}(items)", f"{fn}(items) + {self.operand()}", f"{fn}(items[{self.num(0, 2)}:])", f"{fn}(items) * {self.num(2, 5)}"])
                if not self.has_items:
                    self.has_items = True
                    init = self.rng.choice(["items = []", f"items = [{self.operand()}]", f"items = list(range({self.num(1, 9)}))", f"items = [{self.operand()}, {self.operand()}]"])
                    return [init, add, f"{v} = {use}"]
                return [add, f"{v} = {use}"]
            if fn in ("max", "min", "pow", "divmod"):
                return [f"{v} = {fn}({self.operand()}, {self.num(1, 9)})"]
            return [f"{v} = {fn}({self.expr()})"]

        def s_str(d: int) -> list[str]:
            v = self.new_local()
            form = self.rng.choice([f'"{self.string()}" + str({self.operand()})', f'"{self.string()}".format({self.var()})',
                                    f'f"{self.string()} {{{self.var()}}}"', f'"{self.string()}" * {self.num(1, 3)}', f'"{self.string()}".upper()'])
            return [f"{v} = {form}"] + ([f"if {v}.startswith(\"{self.string()[:3]}\"):", f"{ind}return {self.operand()}"] if self.rng.random() < 0.5 else [])

        def s_dict(d: int) -> list[str]:
            v = self.new_local()
            key = self.rng.choice(self.vocab["nouns"])
            init = self.rng.choice([f'{{"{key}": {self.operand()}, "limit": {self.num()}}}', f'{{"{key}": {self.expr()}}}', f'dict({key}={self.operand()})',
                                    f'{{"{key}": [{self.operand()}], "{self.rng.choice(self.vocab["nouns"])}": None}}'])
            use = self.rng.choice([f'{v}.get("{key}", {self.num()})', f'{v}["{key}"]', f'len({v})', f'{v}.pop("{key}", {self.operand()})'])
            return [f"{v} = {init}", f"{self.new_local()} = {use}"]

        def s_try(d: int) -> list[str]:
            v = self.new_local()
            exc = self.rng.choice(["ZeroDivisionError", "ValueError", "(TypeError, KeyError)", "Exception"])
            body = self.rng.choice([f"{v} = {self.var()} // ({self.operand()} + 1)", f"{v} = int(str({self.operand()}))", f"{v} = {self.expr()}"])
            return ["try:", f"{ind}{body}", f"except {exc}:", f"{ind}{v} = {self.operand()}"]

        def s_comment(d: int) -> list[str]:
            return [f"# {self.rng.choice(self.vocab['verbs'])} {self.rng.choice(self.vocab['nouns'])} ({self.num()})"]

        def s_comp(d: int) -> list[str]:
            v = self.new_local()
            return [f"{v} = sum(x * {self.num(2, 9)} for x in range({self.var()} % {self.num(3, 17)} + 1))"]

        def s_ternary(d: int) -> list[str]:
            v = self.new_local()
            return [f"{v} = {self.operand()} if {self.cond()} else {self.operand()}"]

        kinds: dict[str, Callable[[int], list[str]]] = {
            "assign": s_assign, "update": s_update, "if": s_if, "for": s_for, "while": s_while, "call": s_call,
            "str": s_str, "dict": s_dict, "try": s_try, "comment": s_comment, "comp": s_comp, "ternary": s_ternary,
        }
        body = self._stmts(n, 0, kinds)
        ret = self.rng.choice(self.locals) if self.locals else self.params[0]
        doc = f'"""{self.rng.choice(self.vocab["verbs"]).capitalize()} {self.rng.choice(self.vocab["nouns"])}."""'
        return [f"def {name}({', '.join(self.params)}):", ind + doc] + [ind + s for s in body] + [f"{ind}return {ret}", ""]

    # ------------------------------------------------------------------ c
    def _c(self, name: str, n: int) -> list[str]:
        ind = "    "

        def block(depth: int) -> list[str]:
            return [ind + s for s in self._stmts(self.rng.randint(1, 3), depth + 1, kinds)]

        def s_assign(d: int) -> list[str]:
            v = self.new_local()
            t = self.rng.choice(["int", "unsigned", "long", "uint32_t", "size_t"])
            return [f"{t} {v} = {self.expr()};"]

        def s_update(d: int) -> list[str]:
            if not self.locals:
                return s_assign(d)
            v = self.rng.choice(self.locals)
            return [f"{v} {self.rng.choice(['+=', '-=', '*=', '^=', '|=', '&=', '<<=', '>>='])} {self.rng.choice([self.operand(), self.hexnum()])};"]

        def s_if(d: int) -> list[str]:
            v = self.new_local()
            lines = [f"int {v} = {self.operand()};", f"if ({self.cond()}) {{"] + block(d)
            r = self.rng.random()
            if r < 0.3:
                lines += [f"}} else if ({self.cond()}) {{", f"{ind}{v} = {self.expr()};"]
            if r < 0.7:
                lines += ["} else {", f"{ind}{v} = {self.operand()};"]
            return lines + ["}"]

        def s_for(d: int) -> list[str]:
            head = self.rng.choice([f"for (int i = 0; i < {self.num(2, 40)}; i++) {{", f"for (int i = {self.operand()}; i < {self.num(40, 90)}; i += {self.num(1, 5)}) {{",
                                    f"for (unsigned i = {self.num(1, 9)}; i != 0; i >>= 1) {{", f"for (int i = {self.num(10, 40)}; i > {self.operand()}; i--) {{"])
            return [head] + block(d) + ["}"]

        def s_while(d: int) -> list[str]:
            v = self.new_local()
            init = self.rng.choice([self.num(0, 5), self.operand(), f"{self.var()} % {self.num(2, 7)}"])
            step = self.rng.choice([f"{v} += {self.num(1, 9)};", f"{v} = {v} * 2 + {self.num()};", f"{v} += {self.operand()} + 1;"])
            head = self.rng.choice([f"while ({v} < {self.num(10, 60)}) {{", f"while ({v} < {self.num(10, 60)} && {self.cond()}) {{", f"do {{"])
            tail = ["}"] if not head.startswith("do") else [f"}} while ({v} < {self.num(10, 60)});"]
            return [f"int {v} = {init};", head, f"{ind}{step}"] + block(d) + tail

        def s_call(d: int) -> list[str]:
            v = self.new_local()
            if not self.has_buf:
                self.has_buf = True
                n = self.num(8, 64)
                init = self.rng.choice([f"uint8_t buf[{n}];", f"unsigned char buf[{n}] = {{0}};", f"uint8_t buf[{n}], *bp = buf;"])
                fill = self.rng.choice(["memset(buf, 0, sizeof(buf));", f"memset(buf, {self.operand()} & 0xff, {self.num(1, 8)});",
                                        f"for (size_t j = 0; j < sizeof(buf); j++) buf[j] = (uint8_t)(j ^ {self.operand()});",
                                        f"buf[0] = (uint8_t){self.operand()};"])
                use = self.rng.choice([f"int {v} = buf[{self.num(0, 7)}];", f"int {v} = buf[{self.var()} & 7] + buf[{self.num(0, 7)}];",
                                       f"int {v} = (int)sizeof(buf) - buf[{self.num(0, 7)}];"])
                return [init, fill, f"buf[{self.operand()} & 7] {self.rng.choice(['=', '^=', '+='])} (uint8_t)({self.expr()});", use]
            fn = self.rng.choice(["abs", "strlen", "htonl", "ntohs", "labs"])
            arg = f'"{self.string()}"' if fn == "strlen" else self.expr()
            return [f"int {v} = (int){fn}({arg}){self.rng.choice(['', ' + ' + self.operand(), ' & 0xff'])};"]

        def s_str(d: int) -> list[str]:
            v = self.new_local()
            check = self.rng.choice([f"if ({v}[0] == '{self.rng.choice('abcxyz')}') {{", f"if (strlen({v}) > {self.num(1, 9)}) {{",
                                     f"if ({v}[{self.num(1, 3)}] != '{self.rng.choice('abcxyz')}' && {self.cond()}) {{", f"if (strcmp({v}, \"{self.string()}\") == 0) {{"])
            return [f'const char *{v} = "{self.string()}";', check, f"{ind}return {self.operand()};", "}"]

        def s_switch(d: int) -> list[str]:
            v = self.new_local()
            n_cases = self.rng.randint(1, 4)
            lines = [f"int {v} = {self.operand()};", f"switch ({self.operand()} & {self.num(1, 7)}) {{"]
            for c in range(n_cases):
                lines.append(self.rng.choice([f"case {c}: {v} = {self.expr()}; break;", f"case {c}: {v} += {self.operand()}; break;", f"case {c}:\n{ind}{v} = {self.operand()};\n{ind}break;"]))
            if self.rng.random() < 0.7:
                lines.append(f"default: {v} = {self.operand()}; break;")
            return lines + ["}"]

        def s_comment(d: int) -> list[str]:
            return [f"/* {self.rng.choice(self.vocab['verbs'])} {self.rng.choice(self.vocab['nouns'])} ({self.num()}) */"]

        def s_ternary(d: int) -> list[str]:
            v = self.new_local()
            return [f"int {v} = ({self.cond()}) ? {self.operand()} : {self.operand()};"]

        kinds: dict[str, Callable[[int], list[str]]] = {
            "assign": s_assign, "update": s_update, "if": s_if, "for": s_for, "while": s_while, "call": s_call,
            "str": s_str, "switch": s_switch, "comment": s_comment, "ternary": s_ternary,
        }
        body = self._stmts(n, 0, kinds)
        ret = self.rng.choice(self.locals) if self.locals else self.params[0]
        params = ", ".join(f"{self.rng.choice(['int', 'unsigned', 'long'])} {p}" for p in self.params)
        return [f"static int {name}({params}) {{"] + [ind + s for s in body] + [f"{ind}return (int){ret};", "}", ""]

    # ------------------------------------------------------------------ go
    def _go(self, name: str, n: int) -> list[str]:
        ind = "\t"

        def block(depth: int) -> list[str]:
            return [ind + s for s in self._stmts(self.rng.randint(1, 3), depth + 1, kinds)]

        def s_assign(d: int) -> list[str]:
            v = self.new_local()
            return [f"{v} := {self.expr()}"]

        def s_update(d: int) -> list[str]:
            if not self.locals:
                return s_assign(d)
            v = self.rng.choice(self.locals)
            return [f"{v} {self.rng.choice(['+=', '-=', '*=', '^=', '|=', '&=', '<<=', '>>='])} {self.rng.choice([self.operand(), self.hexnum()])}"]

        def s_if(d: int) -> list[str]:
            v = self.new_local()
            lines = [f"{v} := {self.operand()}", f"if {self.cond()} {{"] + block(d)
            r = self.rng.random()
            if r < 0.3:
                lines += [f"}} else if {self.cond()} {{", f"{ind}{v} = {self.expr()}"]
            if r < 0.7:
                lines += ["} else {", f"{ind}{v} = {self.operand()}"]
            return lines + ["}"]

        def s_for(d: int) -> list[str]:
            if self.has_items and self.rng.random() < 0.4:
                tgt = self.rng.choice(self.locals or self.params)
                head = self.rng.choice(["for _, item := range items {", "for pos, item := range items {", "for item := range items {"])
                return [head, f"{ind}{tgt} {self.rng.choice(['+=', '^=', '|='])} item {self.rng.choice(['+', '*', '-'])} {self.operand()}"] + block(d) + ["}"]
            head = self.rng.choice([f"for i := 0; i < {self.num(2, 40)}; i++ {{", f"for i := {self.operand()}; i < {self.num(40, 90)}; i += {self.num(1, 5)} {{",
                                    f"for i := {self.num(10, 40)}; i > {self.operand()}; i-- {{", f"for i := range make([]int, {self.num(2, 9)}) {{"])
            return [head] + block(d) + ["}"]

        def s_while(d: int) -> list[str]:
            v = self.new_local()
            init = self.rng.choice([self.num(0, 5), self.operand(), f"{self.var()} % {self.num(2, 7)}"])
            step = self.rng.choice([f"{v} += {self.num(1, 9)}", f"{v} = {v}*2 + {self.num()}", f"{v} += {self.operand()} + 1"])
            head = self.rng.choice([f"for {v} < {self.num(10, 60)} {{", f"for {v} < {self.num(10, 60)} && {self.cond()} {{", "for {"])
            body = [f"{ind}{step}"] + block(d)
            if head == "for {":
                body += [f"{ind}if {v} > {self.num(10, 60)} {{", f"{ind}{ind}break", f"{ind}}}"]
            return [f"{v} := {init}", head] + body + ["}"]

        def s_call(d: int) -> list[str]:
            v = self.new_local()
            add = self.rng.choice([f"items = append(items, {self.expr()})", f"items = append(items, {self.operand()}, {self.operand()})",
                                   f"items[0] = {self.operand()}", f"items = append(items[:1], {self.operand()})"])
            use = self.rng.choice([f"len(items)", f"len(items) + {self.operand()}", f"items[len(items)-1]", f"cap(items) * {self.num(2, 5)}"])
            if not self.has_items:
                self.has_items = True
                init = self.rng.choice(["items := []int{}", f"items := []int{{{self.operand()}}}", f"items := make([]int, {self.num(1, 9)})", f"items := []int{{{self.operand()}, {self.operand()}}}"])
                return [init, add, f"{v} := {use}"]
            return [add, f"{v} := {use}"]

        def s_str(d: int) -> list[str]:
            v = self.new_local()
            check = self.rng.choice([f"if len({v}) > {self.num(1, 9)} {{", f"if {v}[0] == '{self.rng.choice('abcxyz')}' {{",
                                     f"if {v} != \"{self.string()}\" && {self.cond()} {{", f"if len({v})%{self.num(2, 5)} == 0 {{"])
            return [f'{v} := "{self.string()}"', check, f"{ind}return {self.operand()}", "}"]

        def s_switch(d: int) -> list[str]:
            v = self.new_local()
            n_cases = self.rng.randint(1, 4)
            lines = [f"{v} := {self.operand()}", f"switch {self.operand()} & {self.num(1, 7)} {{"]
            for c in range(n_cases):
                lines += [self.rng.choice([f"case {c}:", f"case {c}, {c + 10}:"]), f"{ind}{v} {self.rng.choice(['=', '+=', '^='])} {self.expr()}"]
            if self.rng.random() < 0.7:
                lines += ["default:", f"{ind}{v} = {self.operand()}"]
            return lines + ["}"]

        def s_comment(d: int) -> list[str]:
            return [f"// {self.rng.choice(self.vocab['verbs'])} {self.rng.choice(self.vocab['nouns'])} ({self.num()})"]

        def s_buf(d: int) -> list[str]:
            v = self.new_local()
            if not self.has_buf:
                self.has_buf = True
                init = self.rng.choice([f"buf := make([]byte, {self.num(8, 64)})", f"buf := []byte{{{self.num(0, 9)}, {self.num(0, 9)}}}", f"buf := make([]byte, {self.operand()} + {self.num(1, 9)})"])
                return [init, f"buf[{self.operand()}&7] {self.rng.choice(['=', '^=', '+='])} byte({self.expr()} & 0xff)",
                        self.rng.choice([f"{v} := int(buf[{self.num(0, 7)}])", f"{v} := len(buf) + int(buf[{self.num(0, 7)}])", f"{v} := int(buf[0]) << {self.num(1, 7)}"])]
            return [f"buf[{self.num(0, 7)}] {self.rng.choice(['^=', '|=', '+='])} byte({self.operand()})", f"{v} := int(buf[{self.num(0, 7)}]) {self.rng.choice(['+', '*', '^'])} {self.operand()}"]

        def s_ternary(d: int) -> list[str]:
            v = self.new_local()
            return [f"{v} := {self.operand()}", f"if {self.cond()} {{", f"{ind}{v} = {self.operand()}", "}"]

        kinds: dict[str, Callable[[int], list[str]]] = {
            "assign": s_assign, "update": s_update, "if": s_if, "for": s_for, "while": s_while, "call": s_call,
            "str": s_str, "switch": s_switch, "comment": s_comment, "buf": s_buf, "ternary": s_ternary,
        }
        body = self._stmts(n, 0, kinds)
        ret = self.rng.choice(self.locals) if self.locals else self.params[0]
        params = ", ".join(self.params) + " int"
        return [f"func {name}({params}) int {{"] + [ind + s for s in body] + [f"{ind}return {ret}", "}", ""]

    # ------------------------------------------------------------------ javascript
    def _js(self, name: str, n: int) -> list[str]:
        ind = "    "

        def block(depth: int) -> list[str]:
            return [ind + s for s in self._stmts(self.rng.randint(1, 3), depth + 1, kinds)]

        def s_assign(d: int) -> list[str]:
            v = self.new_local()
            return [f"let {v} = {self.expr()};"]

        def s_update(d: int) -> list[str]:
            if not self.locals:
                return s_assign(d)
            v = self.rng.choice(self.locals)
            return [f"{v} {self.rng.choice(['+=', '-=', '*=', '^=', '|=', '&=', '<<=', '>>='])} {self.rng.choice([self.operand(), self.hexnum()])};"]

        def s_if(d: int) -> list[str]:
            v = self.new_local()
            lines = [f"let {v} = {self.operand()};", f"if ({self.cond()}) {{"] + block(d)
            r = self.rng.random()
            if r < 0.3:
                lines += [f"}} else if ({self.cond()}) {{", f"{ind}{v} = {self.expr()};"]
            if r < 0.7:
                lines += ["} else {", f"{ind}{v} = {self.operand()};"]
            return lines + ["}"]

        def s_for(d: int) -> list[str]:
            if self.has_items and self.rng.random() < 0.4:
                tgt = self.rng.choice(self.locals or self.params)
                head = self.rng.choice(["for (const item of items) {", "for (const [pos, item] of items.entries()) {", "for (const item of items.slice(1)) {"])
                return [head, f"{ind}{tgt} {self.rng.choice(['+=', '^=', '|='])} item {self.rng.choice(['+', '*', '-'])} {self.operand()};"] + block(d) + ["}"]
            head = self.rng.choice([f"for (let i = 0; i < {self.num(2, 40)}; i++) {{", f"for (let i = {self.operand()}; i < {self.num(40, 90)}; i += {self.num(1, 5)}) {{",
                                    f"for (let i = {self.num(10, 40)}; i > {self.operand()}; i--) {{", f"for (const i of Array({self.num(2, 9)}).keys()) {{"])
            return [head] + block(d) + ["}"]

        def s_while(d: int) -> list[str]:
            v = self.new_local()
            init = self.rng.choice([self.num(0, 5), self.operand(), f"{self.var()} % {self.num(2, 7)}"])
            step = self.rng.choice([f"{v} += {self.num(1, 9)};", f"{v} = {v} * 2 + {self.num()};", f"{v} += {self.operand()} + 1;"])
            head = self.rng.choice([f"while ({v} < {self.num(10, 60)}) {{", f"while ({v} < {self.num(10, 60)} && {self.cond()}) {{", "do {"])
            tail = ["}"] if head != "do {" else [f"}} while ({v} < {self.num(10, 60)});"]
            return [f"let {v} = {init};", head, f"{ind}{step}"] + block(d) + tail

        def s_call(d: int) -> list[str]:
            v = self.new_local()
            add = self.rng.choice([f"items.push({self.expr()});", f"items.push({self.operand()}, {self.operand()});", f"items.unshift({self.operand()});", f"items[0] = {self.operand()};"])
            use = self.rng.choice(["items.length", f"items.reduce((acc, x) => acc + x, {self.num(0, 9)})", f"items.filter((x) => x > {self.num()}).length",
                                   f"items.map((x) => x * {self.num(2, 9)})[0]", f"items.indexOf({self.operand()})"])
            if not self.has_items:
                self.has_items = True
                init = self.rng.choice(["const items = [];", f"const items = [{self.operand()}];", f"const items = new Array({self.num(1, 9)}).fill({self.num(0, 9)});", f"const items = [{self.operand()}, {self.operand()}];"])
                return [init, add, f"let {v} = {use};"]
            return [add, f"let {v} = {use};"]

        def s_str(d: int) -> list[str]:
            v = self.new_local()
            form = self.rng.choice([f'"{self.string()}" + String({self.operand()})', f'"{self.string()}".repeat({self.num(1, 3)})', f'String({self.expr()})',
                                    f'"{self.string()}".toUpperCase()', f'[{self.operand()}, "{self.string()}"].join("-")'])
            check = self.rng.choice([f"if ({v}.length > {self.num(1, 9)}) {{", f"if ({v}.startsWith(\"{self.string()[:3]}\")) {{", f"if ({v}.length % {self.num(2, 5)} === 0 && {self.cond()}) {{"])
            return [f"const {v} = {form};", check, f"{ind}return {self.operand()};", "}"]

        def s_obj(d: int) -> list[str]:
            v = self.new_local()
            key = self.rng.choice(self.vocab["nouns"])
            init = self.rng.choice([f'{{ {key}: {self.operand()}, label: "{self.string()}" }}', f'{{ {key}: {self.expr()} }}', f'{{ {key}: [{self.operand()}], flag: {self.rng.choice(["true", "false"])} }}',
                                    f'Object.freeze({{ {key}: {self.operand()} }})'])
            use = self.rng.choice([f"{v}.{key} + {self.num()}", f"Object.keys({v}).length", f'{v}["{key}"] | 0', f"({v}.{key} || {self.operand()})"])
            return [f"const {v} = {init};", f"let {self.new_local()} = {use};"]

        def s_math(d: int) -> list[str]:
            v = self.new_local()
            fn = self.rng.choice(["Math.max", "Math.min", "Math.abs", "Math.floor"])
            return [f"let {v} = {fn}({self.var()}, {self.num()});" if fn in ("Math.max", "Math.min") else f"let {v} = {fn}({self.var()} / {self.num(2, 9)});"]

        def s_comment(d: int) -> list[str]:
            return [f"// {self.rng.choice(self.vocab['verbs'])} {self.rng.choice(self.vocab['nouns'])} ({self.num()})"]

        def s_tpl(d: int) -> list[str]:
            v = self.new_local()
            return [f"const {v} = `{self.rng.choice(self.vocab['nouns'])}-${{{self.var()}}}-{self.num()}`;"]

        def s_ternary(d: int) -> list[str]:
            v = self.new_local()
            return [f"let {v} = ({self.cond()}) ? {self.operand()} : {self.operand()};"]

        kinds: dict[str, Callable[[int], list[str]]] = {
            "assign": s_assign, "update": s_update, "if": s_if, "for": s_for, "while": s_while, "call": s_call,
            "str": s_str, "obj": s_obj, "math": s_math, "comment": s_comment, "tpl": s_tpl, "ternary": s_ternary,
        }
        body = self._stmts(n, 0, kinds)
        ret = self.rng.choice(self.locals) if self.locals else self.params[0]
        return [f"function {name}({', '.join(self.params)}) {{"] + [ind + s for s in body] + [f"{ind}return {ret};", "}", ""]


def _func_name(lang: str, verb: str, noun: str, i: int) -> str:
    if lang in ("python", "c"):
        return f"{verb}_{noun}_{i}"
    if lang == "go":
        return f"{verb.capitalize()}{noun.capitalize()}{i}"
    return f"{verb}{noun.capitalize()}{i}"


def _file_header(lang: str, stem: str) -> str:
    return {
        "python": f'"""Module {stem}."""\n\nimport os\n\n\n',
        "c": "#include <stdint.h>\n#include <string.h>\n#include <stdlib.h>\n\n",
        "go": f"package {stem}\n\n",
        "javascript": "'use strict';\n\n",
    }[lang]


def make_source_file(lang: str, kind: str, stem: str, rng: random.Random, n_funcs: int) -> str:
    """Исходный файл с n_funcs функциями; детерминирован по rng."""
    vocab = VOCAB[kind]
    gen = _Gen(lang, rng, vocab)
    parts = [_file_header(lang, stem)]
    for i in range(n_funcs):
        verb, noun = rng.choice(vocab["verbs"]), rng.choice(vocab["nouns"])
        n_stmts = rng.randint(3, 10)  # тела 5–30 строк: есть короткие (бин 0–32 токенов) и длинные
        parts.append(gen.function(_func_name(lang, verb, noun, i + 1), n_stmts))
        parts.append("\n" if lang == "python" else "")
    return "".join(parts)


def make_synthetic_repos(raw_dir: Path, seed: int = SMOKE_SEED, langs: tuple[str, ...] = LANGS,
                         funcs_per_file: tuple[int, int] = (10, 20)) -> dict[str, Path]:
    """Создаёт репозитории REPOS в raw_dir/<split>/<owner__name>/ (с маркером .cloned). Возвращает {split: dir}."""
    raw_dir = Path(raw_dir)
    out: dict[str, Path] = {}
    for split, (dirname, kind) in REPOS.items():
        repo = raw_dir / split / dirname
        for lang in langs:
            rng = random.Random(f"{seed}:{split}:{lang}")
            stem = f"{kind}_{lang[:2]}_core"
            n_funcs = rng.randint(*funcs_per_file)
            text = make_source_file(lang, kind, stem, rng, n_funcs)
            path = repo / FILE_PATHS[lang].format(stem=stem)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        (repo / "README.md").write_text(f"# {dirname}\n\nsynthetic repository ({split}, {kind})\n", encoding="utf-8")
        (repo / ".cloned").write_text("", encoding="utf-8")
        out[split] = repo
    return out


def smoke_overrides(work: Path) -> dict[str, Any]:
    """Переопределения default.yaml для smoke-прогона: пути в work, маленькие k/w, выборки, бутстрэп."""
    work = Path(work)
    return {
        "seed": SMOKE_SEED,
        "paths": {
            "raw_repos": str(work / "raw"), "functions": str(work / "functions"), "queries": str(work / "queries"),
            "indexes": str(work / "indexes"), "results": str(work / "results"), "figures": str(work / "figures"),
            "embeddings": str(work / "embeddings"), "runs": str(work / "runs"),
        },
        "dedup": {"k": 16, "w": 8, "overlap_threshold": 0.3},
        "transforms": {"max_per_split": 24, "partial_lines": [3, 5, 10], "llm_sample_per_split": 8},
        "fingerprint": {"k": 16, "w": 8, "workers": 1, "common_df": 50, "common_file_frac": 0.05, "common_min_files": 3,
                        "exact_window": 5, "minhash_perm": 64, "minhash_shingle": 5, "minhash_lsh_threshold": 0.3},
        "calibration": {"alphas": [0.05, 0.01]},
        "eval": {"bootstrap": 20, "latency_repeats": 3, "latency_threads": [1]},
        "semantic": {"query_batch_size": 8, "combiner_n_index": 100, "combiner_n_anchors": 40, "combiner_n_neg": 40,
                     "ann_top_k": 5, "epochs": 1, "batch_size": 8},
        "privacy": {"attack_leaked_pairs": [0, 20], "bow_vocab": 300, "noise_sigma": [0.0, 0.5], "quant_bits": [8]},
    }


def smoke_config(work: Path, base_config: Path | None = None) -> dict[str, Any]:
    """Конфиг smoke-прогона (dict) поверх configs/default.yaml."""
    from smcode.config import load_config

    return load_config(base_config, overrides=smoke_overrides(Path(work)))


def write_smoke_config(work: Path, base_config: Path | None = None) -> Path:
    """Пишет <work>/smoke.yaml и возвращает путь (ключи '_...' из load_config не сохраняются)."""
    import yaml

    cfg = smoke_config(work, base_config)
    clean = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    path = Path(work) / SMOKE_CONFIG_NAME
    path.write_text(yaml.safe_dump(clean, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def prepare_smoke_workdir(work: Path, seed: int = SMOKE_SEED, base_config: Path | None = None) -> Path:
    """Репозитории + smoke.yaml в work; возвращает путь к конфигу."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    make_synthetic_repos(work / "raw", seed=seed)
    return write_smoke_config(work, base_config)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Синтетические мини-репозитории и smoke-конфиг для стенда.")
    ap.add_argument("--out", required=True, help="рабочий каталог (создаётся): raw/, smoke.yaml, затем functions/ ...")
    ap.add_argument("--seed", type=int, default=SMOKE_SEED)
    ap.add_argument("--base-config", default=None, help="базовый YAML (по умолчанию configs/default.yaml)")
    args = ap.parse_args(argv)
    path = prepare_smoke_workdir(Path(args.out), seed=args.seed, base_config=Path(args.base_config) if args.base_config else None)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
