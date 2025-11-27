import io
try:
    import ifcopenshell
    IFCOPEN_AVAILABLE = True
except Exception:
    IFCOPEN_AVAILABLE = False

def load_ifc_from_bytes(data: bytes):

    if IFCOPEN_AVAILABLE:
        tmp = io.BytesIO(data)
        try:
            return ifcopenshell.open(tmp)
        except Exception:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".ifc", delete=False) as f:
                f.write(data)
                tmp_path = f.name
            return ifcopenshell.open(tmp_path)
    else:
        class MockIFC:
            def by_type(self, t):
                return []
        return MockIFC()
