def build_report(items):
    return {
        "status": "success",
        "instrument_count": len(items),
        "instruments": items
    }
