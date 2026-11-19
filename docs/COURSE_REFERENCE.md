# Verified course reference

The November 16 implementation uses the local professor example at
`C:\MSAIS\EGN6216-AI SYSTEMS\professor_project\DockerFiles`.
Its Dockerfile uses Python 3.8 slim, requirements installation, and `python app.py`.
Its Compose file contains `gradio-app`, Prometheus and Grafana. Its Prometheus
configuration scrapes `gradio-app:8000` every five seconds. Host metrics port
8001 was used to avoid an existing port conflict. The original images use `latest`.
These are verified source settings, not recommended AERIS dependency versions.

The chat **Deploy and monitor Module 8 app** corroborates manual dashboard
creation, a before/after prediction demonstration, and a 12-to-16 prediction
increase. The local notebook and three screenshots remain in professor_project.
No alert rule or Grafana provisioning file was found in that example. The new
AERIS provisioning, persistent volumes, readiness checks and alert rules are
project additions, not attributed to the professor.

AERIS retains the three-service layout, five-second scrape, live request metrics
and before/after validation sequence. It adapts the endpoints, CPU Python 3.11
image, bounded AHU labels, units and local ports. Grafana data source/dashboard
provisioning follows the [official documentation](https://grafana.com/docs/grafana/latest/administration/provisioning/).
Alert pending/firing/recovery behavior follows [Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/).

The master template and AERIS M2/M3/M4 DOCX files were read locally under
`C:\MSAIS\EGN6216-AI SYSTEMS\AHU_PROJECT`. The risk scale is the actual M2
5.2 matrix; three likelihood levels by four impact levels, not an invented
numeric product. Existing R1–R8 identities are preserved in TRUSTWORTHINESS.md.

The user clarified during implementation that the professor's trustworthiness
ideas were general class guidance and no separate resource was retained.
There is no missing-document dependency. Methods are selected for the actual
hybrid and tested end to end: explicit grouped sklearn metrics, raw-window
corruption tests, full-hybrid grouped perturbation, SQLite feedback controls,
and Prometheus/Grafana monitoring. Library names are not treated as a checklist.

Submission cross-check: 3.3 → trustworthiness controls; 5.3 → risk register;
7.6 → experiments/end_to_end_20261116; 8.4–8.5 → deployment configuration and
test evidence; 9.1 → measured targets; 9.2–9.4 → monitoring plan and failure
demonstration; 9.5 → persistent feedback and explicitly separated test cohort.
Historical Word submissions remain unchanged; the current Markdown report is
a dated addendum with reproducible evidence.
