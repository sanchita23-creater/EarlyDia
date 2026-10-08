"""EarlyDia research-only analyses.

ISOLATION RULE: nothing in src/inference, src/app or backend/ may import this package for
prediction. Research code only READS locked artifacts/reports and WRITES to reports/research/.
It never writes to models/ or configs/.
"""
