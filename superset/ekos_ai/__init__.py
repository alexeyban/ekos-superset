# Licensed to the Apache Software Foundation (ASF) under one or more contributor license
# agreements; see the NOTICE file. Licensed under the Apache License, Version 2.0.
"""EKOS additions to Superset: text-to-SQL and text-to-dashboard.

The model only proposes structured terms (a SQL SELECT, or a chart spec). Deterministic code in
``guard`` and ``planner`` decides whether a proposal is accepted, and Superset's own commands
build the datasets, charts and dashboard. Row values are never sent to the model.
"""
