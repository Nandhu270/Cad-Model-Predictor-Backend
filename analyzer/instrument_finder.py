import logging
logger = logging.getLogger("uvicorn.error")

def _lower_or_empty(s):
    try:
        return (s or "").lower()
    except Exception:
        return ""

KEYWORDS = [
    "rotameter", "rotometer", "rotameters",
    "flow meter", "flowmeter", "mag meter", "magnetic flow", "magflow",
    "orifice", "orifice plate", "flow transmitter", "transmitter",
    "flow", "ft", "fm", "sensor", "meter", "rotor"
]

CANDIDATE_IFC_TYPES = [
    "IfcFlowInstrument",
    "IfcPipeSegment",
    "IfcDistributionControlElement",
    "IfcDistributionControlElementType",
    "IfcDistributionPort",
    "IfcFlowSegment",
    "IfcFitting",
    "IfcBuildingElementProxy",
    "IfcElement"
]

def _pset_contains_keyword(el):

    try:
        for rel in getattr(el, "IsDefinedBy", []) or []:
            try:
                pdef = getattr(rel, "RelatingPropertyDefinition", None)
                if pdef and pdef.is_a("IfcPropertySet"):
                    for prop in getattr(pdef, "HasProperties", []) or []:
                        name = _lower_or_empty(getattr(prop, "Name", None))
                        if any(k in name for k in KEYWORDS):
                            return True
                        try:
                            val = getattr(prop, "NominalValue", None)
                            if val:
                                sval = _lower_or_empty(getattr(val, "wrappedValue", None) or str(val))
                                if any(k in sval for k in KEYWORDS):
                                    return True
                        except Exception:
                            pass
            except Exception:
                continue
    except Exception:
        return False
    return False

def find_instruments(ifc):

    candidates = []
    try:
        try:
            i = 0
            for el in ifc.by_type("IfcElement"):
                if i < 200:
                    try:
                        logger.info(f"DEBUG_EL[{i}]: {el.is_a()} | Name={getattr(el,'Name',None)} | Tag={getattr(el,'Tag',None)} | ObjType={getattr(el,'ObjectType',None)}")
                    except Exception:
                        logger.info(f"DEBUG_EL[{i}]: {el.is_a()} (no name/tag)")
                i += 1
                if i >= 200:
                    break
        except Exception:
            logger.info("DEBUG: can't iterate IfcElement for debug printing")

        seen = set()
        for typ in CANDIDATE_IFC_TYPES:
            try:
                for el in ifc.by_type(typ):
                    name = _lower_or_empty(getattr(el, "Name", None))
                    tag = _lower_or_empty(getattr(el, "Tag", None))
                    objt = _lower_or_empty(getattr(el, "ObjectType", None))
                    hay = " ".join([name, tag, objt])

                    matched = any(k in hay for k in KEYWORDS)
                    if not matched:
                        matched = _pset_contains_keyword(el)

                    if matched:
                        uid = getattr(el, "GlobalId", None) or getattr(el, "Tag", None) or id(el)
                        if uid not in seen:
                            candidates.append(el)
                            seen.add(uid)
            except Exception:
                continue

        try:
            for el in ifc.by_type("IfcElement"):
                name = _lower_or_empty(getattr(el, "Name", None))
                tag = _lower_or_empty(getattr(el, "Tag", None))
                objt = _lower_or_empty(getattr(el, "ObjectType", None))
                hay = " ".join([name, tag, objt])
                if any(k in hay for k in KEYWORDS):
                    uid = getattr(el, "GlobalId", None) or getattr(el, "Tag", None) or id(el)
                    if uid not in seen:
                        candidates.append(el)
                        seen.add(uid)
        except Exception:
            pass

        logger.info(f"find_instruments: returning {len(candidates)} candidates")
        return candidates
    except Exception as exc:
        logger.exception("find_instruments failed: %s", exc)
        return []
