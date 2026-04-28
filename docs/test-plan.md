# Test Plan

## Scope

The test plan covers repository-level checks that can be validated without starting the full Docker stack:

1. Documentation presence and completeness.
2. API contract alignment between Streamlit and FastAPI.
3. Endpoint availability in the backend router.
4. Monitoring and deployment wiring in Prometheus and Docker Compose.

## Test Cases

| ID | Area | Expected Result |
| --- | --- | --- |
| TC-01 | Documentation | Architecture, API spec, and test plan files exist and are populated. |
| TC-02 | Backend contract | The dashboard router exposes the history and accuracy endpoints. |
| TC-03 | Frontend contract | Streamlit calls the corrected backend routes and reads compatible fields. |
| TC-04 | Deployment wiring | Docker Compose contains the core services and Prometheus targets. |
| TC-05 | API surface | The prediction endpoint remains available in the gateway service. |

## Acceptance Criteria

- All unit tests pass.
- Core repository contracts remain synchronized across frontend, backend, and deployment files.
- No test depends on live network calls or remote services.

## Execution Strategy

Run the suite from the repository root with:

```bash
python3 -m unittest discover -s tests -v
```
