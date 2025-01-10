"""Module needs to move to a stasndalone package."""
from langchain_core.documents import Document

from langchain_mariadb import FilterExpressionBuilder

metadatas = [
    {
        "name": "adam",
        "date": "2021-01-01",
        "count": 1,
        "is_active": True,
        "tags": ["a", "b"],
        "location": [1.0, 2.0],
        "id": 1,
        "height": 10.0,  # Float column
        "happiness": 0.9,  # Float column
        "sadness": 0.1,  # Float column
    },
    {
        "name": "bob",
        "date": "2021-01-02",
        "count": 2,
        "is_active": False,
        "tags": ["b", "c"],
        "location": [2.0, 3.0],
        "id": 2,
        "height": 5.7,  # Float column
        "happiness": 0.8,  # Float column
        "sadness": 0.1,  # Float column
    },
    {
        "name": "jane",
        "date": "2021-01-01",
        "count": 3,
        "is_active": True,
        "tags": ["b", "d"],
        "location": [3.0, 4.0],
        "id": 3,
        "height": 2.4,  # Float column
        "happiness": None,
        # Sadness missing intentionally
    },
]
texts = ["id {id}".format(id=metadata["id"]) for metadata in metadatas]

DOCUMENTS = [
    Document(page_content=text, metadata=metadata)
    for text, metadata in zip(texts, metadatas)
]


TYPE_1_FILTERING_TEST_CASES = [
    # These tests only involve equality checks
    (
        {"id": 1},
        [1],
    ),
    # String field
    (
        # check name
        {"name": "adam"},
        [1],
    ),
    # Boolean fields
    (
        {"is_active": True},
        [1, 3],
    ),
    (
        {"is_active": False},
        [2],
    ),
    # And semantics for top level filtering
    (
        {"id": 1, "is_active": True},
        [1],
    ),
    (
        {"id": 1, "is_active": False},
        [],
    ),
]

TYPE_2_FILTERING_TEST_CASES = [
    # These involve equality checks and other operators
    # like $ne, $gt, $gte, $lt, $lte
    (
        {"id": 1},
        [1],
    ),
    (
        {"id": {"$ne": 1}},
        [2, 3],
    ),
    (
        {"id": {"$gt": 1}},
        [2, 3],
    ),
    (
        {"id": {"$gte": 1}},
        [1, 2, 3],
    ),
    (
        {"id": {"$lt": 1}},
        [],
    ),
    (
        {"id": {"$lte": 1}},
        [1],
    ),
    # Repeat all the same tests with name (string column)
    (
        {"name": "adam"},
        [1],
    ),
    (
        {"name": "bob"},
        [2],
    ),
    (
        {"name": {"$eq": "adam"}},
        [1],
    ),
    (
        {"name": {"$ne": "adam"}},
        [2, 3],
    ),
    # And also gt, gte, lt, lte relying on lexicographical ordering
    (
        {"name": {"$gt": "jane"}},
        [],
    ),
    (
        {"name": {"$gte": "jane"}},
        [3],
    ),
    (
        {"name": {"$lt": "jane"}},
        [1, 2],
    ),
    (
        {"name": {"$lte": "jane"}},
        [1, 2, 3],
    ),
    (
        {"is_active": {"$eq": True}},
        [1, 3],
    ),
    (
        {"is_active": {"$ne": True}},
        [2],
    ),
    # Test float column.
    (
        {"height": {"$gt": 5.0}},
        [1, 2],
    ),
    (
        {"height": {"$gte": 5.0}},
        [1, 2],
    ),
    (
        {"height": {"$lt": 5.0}},
        [3],
    ),
    (
        {"height": {"$lte": 5.8}},
        [2, 3],
    ),
]

TYPE_3_FILTERING_TEST_CASES = [
    # These involve usage of AND, OR and NOT operators
    (
        {"$or": [{"id": 1}, {"id": 2}]},
        [1, 2],
    ),
    (
        {"$or": [{"id": 1}, {"name": "bob"}]},
        [1, 2],
    ),
    (
        {"$and": [{"id": 1}, {"id": 2}]},
        [],
    ),
    (
        {"$or": [{"id": 1}, {"id": 2}, {"id": 3}]},
        [1, 2, 3],
    ),
    # Test for $not operator
    (
        {"$not": {"id": 1}},
        [2, 3],
    ),
    (
        {"$not": [{"id": 1}]},
        [2, 3],
    ),
    (
        {"$not": {"name": "adam"}},
        [2, 3],
    ),
    (
        {"$not": [{"name": "adam"}]},
        [2, 3],
    ),
    (
        {"$not": {"is_active": True}},
        [2],
    ),
    (
        {"$not": [{"is_active": True}]},
        [2],
    ),
    (
        {"$not": {"height": {"$gt": 5.0}}},
        [3],
    ),
    (
        {"$not": [{"height": {"$gt": 5.0}}]},
        [3],
    ),
]

TYPE_4_FILTERING_TEST_CASES = [
    # These involve special operators like $in, $nin
    # Test in
    (
        {"name": {"$in": ["adam", "bob"]}},
        [1, 2],
    ),
    # With numeric fields
    (
        {"id": {"$in": [1, 2]}},
        [1, 2],
    ),
    # Test nin
    (
        {"name": {"$nin": ["adam", "bob"]}},
        [3],
    ),
    ## with numeric fields
    (
        {"id": {"$nin": [1, 2]}},
        [3],
    ),
]

TYPE_5_FILTERING_TEST_CASES = [
    # These involve special operators like $like, $ilike that
    # may be specified to certain databases.
    (
        {"name": {"$like": "a%"}},
        [1],
    ),
    (
        {"name": {"$like": "%a%"}},  # adam and jane
        [1, 3],
    ),
    (
        {"name": {"$nlike": "a%"}},
        [2, 3],
    ),
    (
        {"name": {"$nlike": "%a%"}},  # adam and jane
        [2],
    ),
]

TYPE_6_FILTERING_TEST_CASES = [
    # These involve the special operator $exists
    (
        {"happiness": {"$exists": False}},
        [],
    ),
    (
        {"happiness": {"$exists": True}},
        [1, 2, 3],
    ),
    (
        {"sadness": {"$exists": False}},
        [3],
    ),
    (
        {"sadness": {"$exists": True}},
        [1, 2],
    ),
]

f = FilterExpressionBuilder()

TYPE_1_EXP_FILTERING_TEST_CASES = [
    # These tests only involve equality checks
    (
        f.eq("id", 1),
        [1],
    ),
    # String field
    (
        # check name
        f.eq("name", "adam"),
        [1],
    ),
    # Boolean fields
    (
        f.eq("is_active", True),
        [1, 3],
    ),
    (
        f.eq("is_active", False),
        [2],
    ),
    # And semantics for top level filtering
    (
        f.both(f.eq("is_active", True), f.eq("id", 1)),
        [1],
    ),
    (
        f.both(f.eq("is_active", False), f.eq("id", 1)),
        [],
    ),
]

TYPE_2_EXP_FILTERING_TEST_CASES = [
    # These involve equality checks and other operators
    # like $ne, $gt, $gte, $lt, $lte
    (
        f.eq("id", 1),
        [1],
    ),
    (
        f.ne("id", 1),
        [2, 3],
    ),
    (
        f.gt("id", 1),
        [2, 3],
    ),
    (
        f.gte("id", 1),
        [1, 2, 3],
    ),
    (
        f.lt("id", 1),
        [],
    ),
    (
        f.lte("id", 1),
        [1],
    ),
    # Repeat all the same tests with name (string column)
    (
        f.eq("name", "adam"),
        [1],
    ),
    (
        f.eq("name", "bob"),
        [2],
    ),
    (
        f.eq("name", "adam"),
        [1],
    ),
    (
        f.ne("name", "adam"),
        [2, 3],
    ),
    # And also gt, gte, lt, lte relying on lexicographical ordering
    (
        f.gt("name", "jane"),
        [],
    ),
    (
        f.gte("name", "jane"),
        [3],
    ),
    (
        f.lt("name", "jane"),
        [1, 2],
    ),
    (
        f.lte("name", "jane"),
        [1, 2, 3],
    ),
    (
        f.eq("is_active", True),
        [1, 3],
    ),
    (
        f.ne("is_active", True),
        [2],
    ),
    # Test float column.
    (
        f.gt("height", 5.0),
        [1, 2],
    ),
    (
        f.gte("height", 5.0),
        [1, 2],
    ),
    (
        f.lt("height", 5.0),
        [3],
    ),
    (
        f.lte("height", 5.8),
        [2, 3],
    ),
]

TYPE_3_EXP_FILTERING_TEST_CASES = [
    # These involve usage of AND, OR and NOT operators
    (
        f.either(f.eq("id", 1), f.eq("id", 2)),
        [1, 2],
    ),
    (
        f.either(f.eq("id", 1), f.eq("name",  "bob")),
        [1, 2],
    ),
    (
        f.both(f.eq("id", 1), f.eq("id", 2)),
        [],
    ),
    (
        f.either(f.either(f.eq("id", 1), f.eq("id", 2)), f.eq("id", 3)),
        [1, 2, 3],
    ),
    # Test for $not operator
    (
        f.negate(f.eq("id", 1)),
        [2, 3],
    ),
    (
        f.negate(f.eq("id", 1)),
        [2, 3],
    ),
    (
        f.negate(f.eq("is_active", True)),
        [2],
    ),
    (
        f.negate(f.gt("height", 5.0)),
        [3],
    ),
]

TYPE_4_EXP_FILTERING_TEST_CASES = [
    # These involve special operators like $in, $nin
    # Test in
    (
        f.includes("name", ["adam", "bob"]),
        [1, 2],
    ),
    # With numeric fields
    (
        f.includes("id", [1, 2]),
        [1, 2],
    ),
    # Test nin
    (
        f.excludes("name", ["adam", "bob"]),
        [3],
    ),
    ## with numeric fields
    (
        f.excludes("id", [1, 2]),
        [3],
    ),
]

TYPE_5_EXP_FILTERING_TEST_CASES = [
    # These involve special operators like $like, $ilike that
    # may be specified to certain databases.
    (
        f.like("name", "a%"),
        [1],
    ),
    (
        f.like("name", "%a%"),
        [1, 3],
    ),
    (
        f.nlike("name", "a%"),
        [2, 3],
    ),
    (
        f.nlike("name", "%a%"),
        [2],
    ),
]
