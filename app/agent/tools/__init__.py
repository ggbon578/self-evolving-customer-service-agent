"""工具注册表。新增工具 → 在这里登记即可被 Agent Loop 使用。"""
from .recall_playbook import RecallPlaybookTool
from .product_search import ProductSearchTool
from .buyer_reviews import BuyerReviewsTool
from .handoff import HandoffTool


def build_registry() -> dict:
    tools = [
        RecallPlaybookTool(),
        ProductSearchTool(),
        BuyerReviewsTool(),
        HandoffTool(),
    ]
    return {t.name: t for t in tools}
