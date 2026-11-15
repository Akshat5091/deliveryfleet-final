# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
Primary: a faculty examiner (CSE3101 Agentic AI, BML Munjal University) watching a live demo on a projector while the four-student group drives the app and explains it; viva questions follow. Secondary: the same examiner opening the deployed link later.

## Product Purpose
DeliveryFleet MAS is a multi-agent dispatch system for urban last-mile delivery across four city zones. A customer message goes through an input guard, a Supervisor agent decomposes the job, and four worker agents (Retrieval/RAG, Dispatch, Route, Exception) decide the driver, route and recovery plan. The website exists to show, run by run, which agent acts where, what it decided, how data moved (Blackboard, MCP tools, A2A messages) and why the result is safe. Success: within about ten seconds of a run the examiner can tell which agent is working now and what was decided; data flow and guardrail verdicts are visible on the same screen.

## Positioning
Every number and event on screen comes from a real run of real agents: CrewAI agents and a Google ADK agent in a separate service, tools reached only through an MCP server, a live trace store. Nothing is a simulated animation.

## Operating Context
Projected laptop screen in a classroom, read from several metres away; students click demo cases and narrate. Also the Runs tab (72 benchmark runs plus live runs, each replayable) and an Evaluation tab of charts used in the report and slides.

## Capabilities and Constraints
- Vanilla HTML/CSS/JS served by FastAPI, no build step; deployed to a Hugging Face Docker Space.
- Live updates via Server-Sent Events; runs take tens of seconds and LLM calls are rate-limited.
- Two frameworks only: CrewAI and Google ADK (course rule).
- Tabs: Dispatch, How it works, Runs, Evaluation, Fleet & data.

## Brand Commitments
Group 8 academic palette is binding for documents and slides (white, ink #111827, crimson #991B1B, forest green #166534, charcoal #1F2937). No emojis. The examiner must believe students built it: no AI-template look, no document look, no flashy effects, not too dense to read on a projector.

## Evidence on Hand
Real data: 10 drivers, 20 orders, 3 policy files, 1,500-row synthetic delay history (labelled synthetic), model card (holdout R2 0.901), run traces in the SQLite store. No customers, testimonials or production claims exist; none may be invented.

## Product Principles
1. Show the agent at work, not a description of it.
2. One run, one screen: decision, active agent, data flow and safety verdict visible together.
3. Readable from the back of a room.
4. Every figure traces to a logged run.
