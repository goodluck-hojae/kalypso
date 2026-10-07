# test_pin_rules.py

from enum import Enum, auto


# ============================================================
# Mock OpBehavior
# ============================================================

class OpBehavior(Enum):
    TUPLE_INDEPENDENT = auto()
    JOIN = auto()
    BLOCKING = auto()


# ============================================================
# Mock Operators
# ============================================================

class BaseOp:
    def __init__(self, behavior):
        self.behavior = behavior
        self.pin = False
        self.unpin = False

    def __repr__(self):
        return f"{self.__class__.__name__}(pin={self.pin}, unpin={self.unpin})"


class SemFilter(BaseOp):
    def __init__(self):
        super().__init__(OpBehavior.TUPLE_INDEPENDENT)


class SemMap(BaseOp):
    def __init__(self):
        super().__init__(OpBehavior.TUPLE_INDEPENDENT)


class SemClassify(BaseOp):
    def __init__(self):
        super().__init__(OpBehavior.TUPLE_INDEPENDENT)


class SemAgg(BaseOp):
    def __init__(self):
        super().__init__(OpBehavior.BLOCKING)


class CartesianProduct(BaseOp):
    def __init__(self):
        super().__init__(OpBehavior.JOIN)


# ============================================================
# Pin/Unpin Logic
# ============================================================

NORMAL = 0
AFTER_CP = 1


def apply_pin_unpin(ops_list):

    for op in ops_list:
        op.pin = False
        op.unpin = False

    state = NORMAL
    chain = []

    def finalize_chain(next_behavior):
        nonlocal chain, state
        if not chain:
            return

        if state == AFTER_CP:
            chain = []
            return

        first = chain[0]
        last = chain[-1]

        if next_behavior == OpBehavior.JOIN:
            first.pin = True

        elif next_behavior == OpBehavior.BLOCKING:
            first.pin = True
            last.unpin = True

        elif next_behavior is None:
            if len(chain) > 1:
                first.pin = True
                last.unpin = True

        chain = []

    n = len(ops_list)

    for i, op in enumerate(ops_list):
        next_op = ops_list[i + 1] if i + 1 < n else None
        next_behavior = next_op.behavior if next_op else None

        if op.behavior == OpBehavior.TUPLE_INDEPENDENT:
            chain.append(op)

            if next_op is None or next_behavior != OpBehavior.TUPLE_INDEPENDENT:
                finalize_chain(next_behavior)

        elif op.behavior == OpBehavior.JOIN:
            state = AFTER_CP
            chain = []

        elif op.behavior == OpBehavior.BLOCKING:
            state = NORMAL
            chain = []

        else:
            raise ValueError("Unknown OpBehavior")

    return ops_list


# ============================================================
# Utilities
# ============================================================

def flags(ops):
    return [(op.__class__.__name__, op.pin, op.unpin) for op in ops]


def run_case(name, ops, expected):
    apply_pin_unpin(ops)
    result = flags(ops)

    print(name, "->", result)

    assert result == expected, f"{name} FAILED\nExpected {expected}\nGot {result}"


# ============================================================
# Tests (12 Cases)
# ============================================================

def main():

    cases = [

        # 1
        ("Case 1",
         [SemFilter()],
         [("SemFilter", False, False)]),

        # 2
        ("Case 2",
         [SemFilter(), SemMap()],
         [("SemFilter", True, False),
          ("SemMap", False, True)]),

        # 3
        ("Case 3",
         [SemFilter(), SemMap(), SemClassify()],
         [("SemFilter", True, False),
          ("SemMap", False, False),
          ("SemClassify", False, True)]),

        # 4
        ("Case 4",
         [SemFilter(), SemMap(), SemClassify(), SemAgg()],
         [("SemFilter", True, False),
          ("SemMap", False, False),
          ("SemClassify", False, True),
          ("SemAgg", False, False)]),

        # 5
        ("Case 5",
         [SemMap(), CartesianProduct(), SemFilter()],
         [("SemMap", True, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False)]),

        # 6
        ("Case 6",
         [SemMap(), CartesianProduct(), SemFilter(), SemAgg()],
         [("SemMap", True, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False),
          ("SemAgg", False, False)]),

        # 7
        ("Case 7",
         [SemFilter(), SemMap(), CartesianProduct(), SemFilter()],
         [("SemFilter", True, False),
          ("SemMap", False, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False)]),

        # 8
        ("Case 8",
         [SemMap(), CartesianProduct(), SemFilter(), SemMap()],
         [("SemMap", True, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False),
          ("SemMap", False, False)]),

        # 9
        ("Case 9",
         [SemMap(), CartesianProduct(), SemFilter(), SemAgg(),
          SemFilter(), SemMap()],
         [("SemMap", True, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False),
          ("SemAgg", False, False),
          ("SemFilter", True, False),
          ("SemMap", False, True)]),

        # 10
        ("Case 10",
         [SemFilter(), SemMap(), CartesianProduct(), SemFilter(),
          SemAgg(), SemFilter()],
         [("SemFilter", True, False),
          ("SemMap", False, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False),
          ("SemAgg", False, False),
          ("SemFilter", False, False)]),

        # 11
        ("Case 11",
         [SemFilter(), SemMap(), CartesianProduct(), SemFilter(),
          SemAgg(), SemFilter(), SemMap()],
         [("SemFilter", True, False),
          ("SemMap", False, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False),
          ("SemAgg", False, False),
          ("SemFilter", True, False),
          ("SemMap", False, True)]),

        # 12
        ("Case 12",
         [SemMap(), CartesianProduct(), SemFilter(), SemAgg(),
          SemFilter(), SemMap(), SemClassify()],
         [("SemMap", True, False),
          ("CartesianProduct", False, False),
          ("SemFilter", False, False),
          ("SemAgg", False, False),
          ("SemFilter", True, False),
          ("SemMap", False, False),
          ("SemClassify", False, True)]),
    ]

    for name, ops, expected in cases:
        run_case(name, ops, expected)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()




        # #todo "parse it"
        # # Filter-Filter
        # ff_operators = (
        #     ops.SemFilter("The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie", pin=True),
        #     ops.SemFilter("The review criticizes the movie’s plot, storytelling, or narrative structure"),
        # )
        # # Filter-Filter-Map
        # ffm_operators = (
        #     ops.SemFilter("The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie", pin=True),
        #     ops.SemFilter("The review criticizes the movie’s plot, storytelling, or narrative structure"),
        #     ops.SemMap("Summarize the review"),
        # )
 
        # # Map-Filter-Filter
        # mff_operators = (
        #     ops.SemMap("Summarize the review", pin=True),
        #     ops.SemFilter("The review contains substantive content, meaningful or vague and expresses a concrete opinion about the movie"),
        #     ops.SemFilter("The review criticizes the movie’s plot, storytelling, or narrative structure", unpin=True),
        # )

        # # Filter - GroupBy - Aggregation
        # groups = ["Positive", "Negative"]
        # fga_operators = (
        #     ops.SemFilter("The review contains substantive content, meaning it is not short (less than three sentences) or vague and expresses a concrete opinion about the movie", pin=True),              
        #     ops.SemClassify(groups, unpin=True),   
        #     ops.SemAgg("Find the common opinion"),
        # )

        # # Join-Filter
        # research_categories = data.research_category_data()
        # jf_operators = (
        #     ops.SemMap("Summarize the research abstract and explain how it is related to the category", pin=True),
        #     ops.CartesianProduct(right_table=research_categories),
        #     ops.SemFilter("Is the research paper related to the given category?"),
        #     ops.SemAgg("Find the common opinion"),
        # )
        # return jf_operators