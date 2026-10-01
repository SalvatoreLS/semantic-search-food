from pydantic import BaseModel, ConfigDict, Field

Interval = list[float | None]


class Source(BaseModel):
    label: str
    rank: int


class ItemCard(BaseModel):
    item_id: str
    name: str
    category_path: str
    description: str
    price: float
    price_bucket: str
    attributes: list[str]
    is_food: bool


class Result(ItemCard):
    rank: int
    score: float
    sources: list[Source]
    judge_grade: int | None
    judge_reason: str | None
    doc_text: str


class QueryMetrics(BaseModel):
    ndcg5: float | None
    ndcg10: float | None
    p5: float | None
    food_leak5: float | None


class QueryRef(BaseModel):
    query_id: str
    text: str


class SystemInfo(BaseModel):
    id: str
    label: str
    short: str
    desc: str


class QueriesResponse(BaseModel):
    queries: list[QueryRef]
    systems: list[SystemInfo]
    judge_model: str


class FoodPrior(BaseModel):
    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

    on: bool
    lambda_: float = Field(alias="lambda")
    reason: str


class Stage(BaseModel):
    name: str
    model: str | None
    detail: str | None
    ms: float
    cost_usd: float


class SearchResponse(BaseModel):
    query_id: str | None
    query: str
    system: str
    query_type: str | None
    intent: str | None
    expanded_dishes: list[str]
    food_prior: FoodPrior | None
    understanding_model: str | None
    stages: list[Stage]
    results: list[Result]
    metrics: dict[str, QueryMetrics] | None


class CompareSide(BaseModel):
    system: str
    results: list[Result]
    metrics: QueryMetrics | None


class CompareResponse(BaseModel):
    query: str
    query_type: str | None
    intent: str | None
    left: CompareSide
    right: CompareSide


class LabelPair(BaseModel):
    pair_id: str
    query: str
    item: ItemCard


class LabelProgress(BaseModel):
    done: int
    total: int


class LabelQueue(LabelProgress):
    pairs: list[LabelPair]


class LabelIn(BaseModel):
    pair_id: str
    grade: int = Field(ge=0, le=3)


class SummaryRow(BaseModel):
    system: str
    label: str
    desc: str
    ndcg5: Interval | None
    ndcg10: Interval | None
    p5: Interval | None
    food_leak5: Interval | None


class JudgeSummary(BaseModel):
    model: str
    kappa: float | None
    judged: int
    unjudged: int


class Summary(BaseModel):
    n_queries: int
    rows: list[SummaryRow]
    judge: JudgeSummary
    cost_usd: float | None
    label: LabelProgress
