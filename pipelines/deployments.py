"""Create/update production Prefect deployments."""

from pathlib import Path

from prefect.schedules import Cron

from pipelines.monitoring_flow import monitoring_pipeline
from pipelines.reference_flow import reference_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_POOL = "open-food-mlops"


def main() -> None:
    # Reference generation is intentionally NOT scheduled.
    #
    # It should be run when establishing or deliberately refreshing
    # the monitoring baseline.
    reference_pipeline.from_source(
        source=str(PROJECT_ROOT),
        entrypoint="pipelines/reference_flow.py:reference_pipeline",
    ).deploy(
        name="reference-generation",
        work_pool_name=WORK_POOL,
        parameters={"refresh_source": False},
    )

    # Monitoring runs every 6 hours.
    monitoring_pipeline.from_source(
        source=str(PROJECT_ROOT),
        entrypoint="pipelines/monitoring_flow.py:monitoring_pipeline",
    ).deploy(
        name="production-monitoring",
        work_pool_name=WORK_POOL,
        schedules=[
            Cron(
                "0 */6 * * *",
                timezone="Africa/Khartoum",
                slug="every-6-hours",
            )
        ],
        concurrency_limit=1,
    )

    print("Prefect deployments created/updated.")


if __name__ == "__main__":
    main()
