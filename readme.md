# GigShield - An Autonomous Parametric Income Insurance for Gig Workers

Prototype Demo: [https://youtu.be/hLz2ZEhvils](https://youtu.be/hLz2ZEhvils)

Solution explanation + Financial model + Technical architecture: [https://youtu.be/8LG6JR4Q7rM](https://youtu.be/8LG6JR4Q7rM)

## 1. Project Aim

The primary objective of this project is to provide income insurance for gig workers, protecting them against income loss caused by external disruptions. To achieve this, we are developing a fully autonomous Parametric Insurance Model.

**Persona chosen:** Quick-Commerce Businesses

## 2. Target Market

- **Primary Demographic:** Gig workers in India (projected to reach approximately 23 million by 2030).
- **Economic Scale:** The Indian gig economy is projected to be valued at $455 billion by FY2029-30.

## 3. Business & Operational Strategy

### Operating Profit Expectation

Our target operating profit margin is approximately 15%. This benchmark aligns with the average profit margins maintained by established insurance providers in the nation.

### Contract Service Model

Our organization functions as a B2B2C service, partnering directly with various gig service-providing clients (e.g., Zepto, Blinkit).

- **Dynamic Costing:** Our upfront costs will be dynamically adjusted for different clients based on their total number of workers and specific monetization options.
- **Profit Margin Maintenance:** Our costing structure is designed to strictly maintain the 15% operating profit margin. This is achieved by ensuring that the maximum potential insurance rollout and other operational costs do not exceed our defined profit limits.
- **Data Sharing Agreements:** The upfront partnership agreements will include mandatory Terms and Conditions (T&C) for the secure sharing of client inventory data, worker databases, and employee activity logs.
- **Adaptability:** Based on evolving market dynamics, we will continuously alter our operational conditions to ensure sustained profitability.

### Serviceable Available Market (SAM)

This represents a first-of-its-kind insurance system in India.

- **First-Mover Advantage:** Being the first to enter this space, our goal is to tap into the entire gig ecosystem.
- **Pricing Leverage:** Entering the market without direct precedent allows us to establish the initial premium pricing structures.
- **Market Capture:** Our initial strategy leverages this unique market opportunity to capture maximum market share. As market dynamics shift over time, our pricing system is designed to adapt dynamically.

## 4. Benefits to Society

This model provides crucial social security to a highly uncertain gig economy.

- **Economic Protection:** The gig economy, despite its projected $455 billion valuation by FY2030, currently operates on the verge of collapse when faced with even a single external uncertainty.
- **Livelihood Security:** Our model provides tangible income security to gig workers, thereby protecting a massive and vulnerable demographic that serves as a major contributor to the national economy.

## 5. Categorization of Parametric Disruptions

The insurance model is engineered to trigger autonomously when gig workers experience a quantifiable loss of income resulting from specific, pre-defined external disruptions. These parametric triggers are strictly categorized into the following risk sectors:

- **Adverse Weather and Meteorological Events:**
    - **Precipitation:** Severe or heavy rainfall that impedes navigation and delivery operations.
    - **Thermal Extremes:** Very high or very low temperatures that pose occupational hazards or reduce operational efficiency.
    - **Severe Storms:** Cyclones, typhoons, and localized storms that prompt official meteorological alerts and mandate the cessation of gig activities.
- **Societal and Civic Disruptions:**
    - **Labor Actions:** Organized strikes that halt transit or supply chain logistics.
    - **Civil Restrictions:** Government-mandated curfews that legally prevent workers from operating during standard hours.
    - **Public Gatherings:** Large-scale rallies or protests that block critical infrastructure or cause widespread geographic routing failures.
- **Large-Scale Infrastructural and Technical Failures:**
    - **Resource Scarcity:** Severe shortages of fuel that physically prevent motorized gig workers from fulfilling active orders.
    - **Telecommunications Outages:** Widespread network downtime or internet service provider (ISP) blackouts.
    - **Platform-Specific Technical Glitches:** Internal application failures, server downtimes, or API crashes originating from the partner platform (e.g., the delivery app itself going offline).

## 6. Business Architecture & Data Integration Pipeline

We establish direct technological and operational partnerships with gig economy aggregators (e.g., Zepto, Blinkit).

To ensure accuracy in dynamic pricing and to facilitate instantaneous, fraud-proof insurance rollouts, our integration agreements mandate the secure, continuous sharing of the following partner data streams:

- **Real-Time Inventory Levels:** Analyzed to cross-verify the actual impact of localized disruptions (e.g., if a dark store cannot dispatch goods, it corroborates the worker's inability to earn).
- **Comprehensive Worker Databases:** Required to establish baseline financial profiles, specifically to calculate the historical average income of each individual worker, which serves as the foundation for premium calculation.
- **Granular Activity Logs:** Continuous ingestion of worker GPS coordinates, active login hours, and order delivery histories. This data is critical for validating the worker's presence in a disrupted zone and forms the basis of our fraud-detection mechanisms.

## 7. Eligibility Criteria & Onboarding Validation

During the initial user onboarding phase, the system conducts an evaluation to classify the employment status of the gig worker.

- **Covered Demographics:** The insurance schemes are exclusively applicable to:
    - Full-time delivery partners.
    - Part-time delivery partners who possess no alternative means of external income.
- This rigorous eligibility check ensures that the risk pool consists only of individuals whose primary livelihood is genuinely threatened by gig-economy disruptions.

## 8. Dynamic Pricing Architecture & Insurance Slabs

The platform abandons traditional static insurance premiums in favor of a high-frequency, dynamic weekly pricing model.

- **The Weekly Cycle:** At the beginning of every week, the engine processes new situational parameters, generates a newly optimized upfront agreement, and recalibrates the dynamic pricing for each individual user based on forecasted risks and historical behavior.
- **Base Premium Calculation:** The foundational insurance premium is standardized at **4% of the worker's average weekly earnings**. To ensure accuracy and account for seasonality, this average is calculated using a rolling mean of the worker's income over the preceding 52 weeks and also based on the disruption predicted for the upcoming week.
- **Tiered Coverage Options (Slabs):** Workers are provided with three customized coverage slabs, allowing them to balance their premium costs against their required security net:
    - **Slab 1 (50% Coverage):** Designed for maximum affordability. The premium is set at **3.6%** of the weekly average income (representing a 10% reduction from the standard premium).
    - **Slab 2 (75% Coverage):** The standard tier. The premium operates exactly at the base rate of **4.0%** of the weekly average income.
    - **Slab 3 (100% Coverage):** Designed for comprehensive income protection. The premium is set at **4.8%** of the weekly average income (representing a 20% increase from the standard premium).

## 9. Predictive Premium Calculation & Actuarial Variables

The core predictive model dynamically recalculates the premium every week by evaluating past behavioral data, current environmental states, and future forecasts.

To ensure precision and financial viability, the model analyzes the following statistical and predictive variables:

- **Historical Baselines:** The 52-week moving average of the worker's weekly income, combined with an analysis of their highest and lowest earning weeks.
- **Statistical Variance:** The standard deviation of income and its correlation coefficients with various external factors.
- **Predictive Seasonality & Trend Modeling:** Premiums are dynamically scaled against seasonal income trends. During periods of historically low income—when a worker's financial capacity to withstand a disruption is lowest—the premium is adjusted upward. Conversely, during high-income periods, the premium is reduced.
- **Risk-Based Forecasting:** If external monitoring APIs predict an upcoming disruption (e.g., severe floods expected next week), the risk assessment engine autonomously increases the premium for that cycle.
- **Loyalty & Flexibility:** Users enrolled continuously from the beginning of a cycle maintain baseline plans. However, the system includes provisions for users to switch coverage slabs, with the new dynamic premium weighted mathematically against their past premium payment history.

## 10. Rewards, Penalties, & Anti-Exploitation Protocols

The system enforces strict financial discipline through an automated incentive and penalty structure:

- **Consistency Reward:** For every 4 consecutive weeks of successful premium payment, the worker receives a **0.5% reduction** in their premium, while maintaining the exact same coverage level.
- **Default Penalty:** A comprehensive 52-week default analysis is conducted. For every week a premium payment is missed, a strict **2% penalty** is added to future premiums.
- **8-Week Cooling Period:** To entirely eliminate adverse selection (the practice of workers enrolling only days before a forecasted disruption, such as a cyclone), mid-cycle enrollees are subjected to a mandatory 8-week cooling period. A worker must successfully pay premiums for a minimum of 8 weeks before they are eligible to receive a claim.
- **Fraud Mitigation:** Applications attempting to exploit the platform by securing insurance solely during times of pre-predicted adversities are autonomously rejected by the system. Furthermore, detected fraudulent activity (like GPS spoofing) drastically impacts the user's fraud rating, directly increasing their premium costs.

## 11. Autonomous Insurance Roll-out & The 75% Income Threshold

The payout sequence operates completely autonomously across predefined systemic layers without human intervention.

### The Escalation Workflow

1. **Layer 1:** Detects the trigger point of an external disruption via real-time monitoring.
2. **Layer 2 (Validation & Calculation):** Upon verifying the severity of the localized disruption, the engine cross-references the geographic data of the workers to conduct a localized fraud check. It then calculates the exact coverage percentage required based on the disruption tier, checks the worker's specific pricing slab, and initiates the automated rollout.

### The 75% Actuarial Logic

A payout is solely triggered if the disruption causes the worker's income to drop below **75% of their 52-week moving average** (representing a net loss exceeding 25%).

- **Why 25%?** Gig economy earnings are inherently cyclic. Statistical modeling indicates that an income variance of up to 25% aligns with standard deviations and can typically be recovered by the worker in subsequent operational cycles.
- **Abnormal Loss Coverage:** The model only covers *abnormal* losses that breach this standard deviation. If an external disruption occurs, but the worker's net income loss is 25% or less, no payout is initiated. The maximum coverable amount is exclusively the financial difference between the actual earned income and the 75% threshold limit.

## 12. Parametric Disruption Thresholds

Insurance payouts strictly depend on the validated severity of the external event: (Currently identified parameters for rain, will develop the parameters for other disruptions in the course of building)

- **Precipitation (Heavy Rain):**
    - Less than 5cm: No trigger
    - 5cm to 10cm: 40% coverage
    - 10cm to 15cm: 60% coverage
    - 15cm to 20cm: 80% coverage
    - 20cm and above: 100% coverage
- **Thermal Extremes (Heat/Cold):** Trigger thresholds dynamically calibrate depending on the historical climate data of the specific region.
- **Cyclones & Landslides:** Payouts scale based on the severity of official alerts issued by regional meteorological departments and the exact coordinates of structural impacts.
- **Societal & Technical Issues:** Scaled proportionately based on the precise number of hours or days the operational service or region was disabled.

## 13. Calculated Payout Scenarios (Case Studies)

### Scenario 1: Normal Condition

- **Moving Average Income:** Rs. 8000
- **Base Premium:** 4% of Rs. 8000 = Rs. 320
- **Plan Selected:** Slab 3 (100% coverage)
- **Premium amount:** 4.8% of Rs. 8000 = Rs. 384 (due to slab 3)
- **Claim Threshold (75% of 8000):** Rs. 6000
- **Week’s Actual Income:** Rs. 4500
- **Net Coverable Loss:** Rs. 1500
- **Disruption:** Heavy Rains (15cm) triggering 80% coverage eligibility
- **Calculation:** 80% of the Rs. 1500 loss = Rs. 1200. Because the user is on Slab 3, 100% of this eligible amount is covered.
- **Final Roll-out Amount: Rs. 1200**

### Scenario 2: Prior Defaulter

- **Moving Average Income:** Rs. 8000
- **Plan Selected:** Slab 3 (100% coverage)
- **Premium amount:** 4.8% of Rs. 8000 = Rs. 384
- **Infraction:** Defaulted 6 weeks in the past 52-week cycle
- **Penalty Calculation:** 2% penalty * 6 weeks = 12%
- **Net Premium Paid:** Rs 384 + 12% of Rs. 384 = Rs. 430
- **Claim Threshold (75% of 8000):** Rs. 6000
- **Week’s Actual Income:** Rs. 4500
- **Gross Coverable Loss:** Rs. 1500
- **Disruption:** Heavy Rains (15cm) triggering 80% coverage eligibility
- **Adjusted Loss (after default fines applied to coverage limits):** Rs. 1320
- **Calculation:** 80% of Rs. 1320
- **Final Roll-out Amount: Rs. 1056**

### Scenario 3: Consistent Non-Defaulter

- **Moving Average Income:** Rs. 8000
- **Plan Selected:** Slab 3 (100% Coverage)
- **Initial Premium:** Rs. 384
- **Payment Record:** Paid without fault for 52 consecutive weeks.
- **Premium Reward Calculation:** A 0.5% premium reduction is earned for every 4-week period of consistent payment.
    - *Net Premium Reward:* A reduction of Rs. 42.
- **Net Premium Paid:** Rs. 342 (Initial Rs. 384 - Rs. 42 reward)
- **Claim Threshold (75% of 8000):** Rs. 6000
- **Actual Weekly Income:** Rs. 4500
- **Gross Coverable Loss:** Rs. 1500 (Threshold Rs. 6000 - Actual Rs. 4500)
- **Disruption Factor:** Heavy Rains (15cm), which triggers an 80% baseline coverage payout.
- **Coverage Reward (Loyalty Bonus):** An additional 13% bonus (0.5% accrued per 4 weeks over 52 weeks) is applied directly to the gross coverable loss limit.
- **Adjusted Coverable Loss:** Rs. 1695 (Gross loss of Rs. 1500 + 13% loyalty bonus)
- **Final Payout Calculation:** 80% of Rs. 1695 = Rs. 1356 (weather disruption coverage applied to the adjusted coverable loss).
- **Final Roll-out Amount: Rs. 1356**

## 14. Fraud Detection & Risk Mitigation Mechanisms

To maintain the integrity of the risk pool, the engine utilizes a continuous, multi-layered fraud detection system:

- **GPS & Activity Validation:** Actively monitors for GPS spoofing. If a worker fails to resume active status immediately upon the resolution of a disruption, they are excluded from the insurance rollout for that cycle.
- **Cohort Cross-Validation:** The system cross-references the activity logs of all workers affiliated with a specific geographic outlet or dark store. If over 50% of the localized workforce is unable to recover operations post-disruption, the event is validated. Conversely, if a smaller proportion remains inactive while the majority resumes work, the inactive users are flagged for potential fraud.
- **Behavioral Tracking:** Continuous ingestion of operational metrics (e.g., order acceptance/decline rates, total logged work hours). Exceptionally high decline rates coupled with low active hours automatically trigger a system flag.
- **Dynamic Trust Rating:** Workers are assigned a systemic fraud trust rating ranging from 0.0 (High Fraud Risk) to 5.0 (Legitimate/No Fraud).
    - If a rating falls below 2.5, the user is penalized with a baseline 1% extra premium.
    - For every subsequent 0.1 reduction in the rating below 2.5, an additional 0.1% is dynamically added to their premium.

## 15. Predictive Modeling & Explainable AI (XAI)

The system moves beyond traditional black-box actuarial tables by deploying a robust, highly accurate predictive model driven by Explainable AI (XAI). This ensures complete transparency, allowing the system to justify every calculated number and maintain trust with users.

- **Algorithmic Engine:** Integrates feature engineering powered by advanced statistical measures and ensemble stacking.
- **Dynamic Premium Calculation:** The model recalculates the premium and net coverage dynamically every week by evaluating past data, real-time current scenarios, and future forecasting.
- **Statistical Inputs:** Computations incorporate moving averages of income, correlation coefficients, standard deviations, default rates, on-time payment metrics, and Z-scores.
- **Risk Assessment Integration:** Past statistical scores are merged with real-time situational monitoring and predictive seasonality trends to autonomously alter the upcoming week's premium.

## 16. Autonomous Multi-Agent AI System Architecture

The platform is powered by an autonomous, multi-agent artificial intelligence ecosystem. By converging Retrieval-Augmented Generation (RAG) with the Model Context Protocol (MCP), the system achieves a powerful hybrid intelligence capable of making context-aware, real-time decisions without human intervention.

### High-Level System Workflow

```
External APIs → MCP Layer → Trigger Monitoring Agent → Validation Agent 
→ RAG Layer (Vector DB) → Parallel Decision Agents → Decision Orchestrator 
→ SQL Layer → ML Predictive Model → Final Output / Roll-out
```

### The Four-Layer Core Architecture

1. **Layer 1 (Monitoring):** Utilizes API integrations to continuously track external factors against predefined trigger thresholds in real-time. Once a threshold is breached, the workflow advances.
2. **Layer 2 (Validation):** Validates the exact physical extent of the disruption, verifies worker location accuracy, checks active app status during the disruption timeframe, and categorizes the severity of the event.
3. **Layer 3 (Actuarial Assessment):** Calculates the financial impact. It verifies if the income loss exceeds the 25% cyclic threshold (against the 52-week moving average), checks the worker's selected insurance slab, and applies all relevant rewards or penalties.
4. **Layer 4 (Execution):** Automatically finalizes and rolls out the calculated insurance payout.

### High level Architecture

```
External Systems (APIs, Streams)
            ↓
        MCP Layer
            ↓
   Trigger Monitoring Agent
            ↓
     Validation Agent
            ↓
        RAG Layer
            ↓
   Parallel Decision Agents
            ↓
     Decision Orchestrator
            ↓
        SQL Layer
            ↓
 Output Storage + ML Pipeline
```

### Agent Framework & Core Components

Developed utilizing the LangChain/Ollama framework, the ecosystem is composed of several specialized agents:

- **MCP (Model Context Protocol):** Connects agents to live, real-time external data (e.g., Weather APIs, News APIs, transactional streams). Agents can also execute autonomous web crawling to identify the impact of emerging conditions.
- **RAG (Retrieval-Augmented Generation):** Queries Vector Databases (Pinecone/Chroma) to retrieve semantic knowledge, historical cases, domain rules, and fraud patterns.
- **SQL Database Layer:** Serves as the structured transactional memory, storing agent outputs, user histories, and system state logs for downstream ML pipeline enrichment.
- **Purpose:**
    - Store:
        - Agent outputs
        - Decisions
        - Logs
        - System state
    - Provide:
        - Structured queries
        - Historical records
        - Transactional consistency

# 17. End-to-End Workflow

## Step 1: Continuous Monitoring

### Agent: MonitorAgent

- Uses MCP to fetch live data
- Runs continuously
- Detects trigger conditions

**Flow:**

```
Loop:
   Fetch real-time data via MCP
   Evaluate trigger rules
   If condition met → emit event
```

## Step 2: Trigger Validation

### Agent: ValidationAgent

- Re-validates trigger using MCP
- Ensures accuracy and consistency

**Key Role:**

- Prevent false positives
- Confirm threshold conditions

## Step 3: Context Enrichment via RAG

### Agent: ContextAgent

- Queries vector DB using semantic search
- Retrieves relevant historical knowledge

**Output:**

- Context-enriched input for downstream agents

## Step 4: Parallel Multi-Agent Processing

### Agents:

- FraudDetectionAgent
- RuleValidationAgent
- RiskScoringAgent

**Execution:**

- Runs in parallel (async / distributed)

**Each agent receives:**

- MCP data (real-time)
- RAG data (historical)
- Trigger context

**Processing:**

- Analyze independently
- Generate structured outputs

## Step 5: Decision Orchestration

### Agent: DecisionAgent

- Aggregates outputs from all agents
- Applies dynamic rules
- Determines final action

**Possible Outcomes:**

- Approve
- Flag
- Escalate
- Reject

## Step 6: SQL Enrichment

### Agent: SQLAgent

- Queries SQL DB for:
    - User history
    - Activity logs
    - Past decisions
- Combines structured + inferred data

## Step 7: Output Handling

- Final output is:
    - Stored in SQL / output DB
    - Passed to ML pipeline for Premium calculation and Coverage calculations

# 18. MARKET CRASH: Adversarial Defense & Anti-Spoofing Strategy

## Objective

To ensure that insurance payouts are granted **only to genuinely impacted workers**, by detecting and preventing **location spoofing, intentional inactivity, and coordinated fraud**.

## Core Principle

We do not rely on self-reported GPS alone.

Instead, we validate claims using:

> **Proof of Presence + Proof of Work + Proof of Impact**
> 

A worker is eligible **only if all three conditions are satisfied**:

- Presence in disruption zone
- Active participation in work ecosystem
- Measurable income loss

## Threat Model

Potential fraud strategies include:

- Spoofing GPS to appear in affected zones
- Staying inactive intentionally during disruptions
- Jumping into high-risk zones only during trigger windows
- Coordinated fraud by multiple workers
- Manipulating activity logs before/after disruption

## Multi-Layer Anti-Spoofing System

### 1. Work-Anchored Location Validation

We prioritize **economic activity signals over raw GPS**.

**Signals used:**

- Order pickup/drop locations
- Delivery routes
- Time spent near assigned outlet

Eligibility requires **real work activity** in the disruption window

No orders → No claim eligibility

### 2. Disruption Window Presence Score

Each worker is assigned a **Presence Score**:

$$
Presence Score = Active time in zone + Orders completed + Time near outlet
$$

Flag if:

- GPS shows presence
- BUT no movement / no orders / no outlet interaction

### 3. Movement Realism Engine

We validate whether movement patterns resemble **real delivery behavior**.

**Expected:**

- Non-linear routes
- Stop points (restaurants, deliveries)
- Variable speeds

Fraud Indicators:

- Teleportation between locations
- Perfect straight-line movement
- Static coordinates during active hours

### 4. Outlet Gravity Model

Each worker is expected to show **behavior centered around assigned outlets**.

**Checks:**

- Distance from outlet over time
- Frequency of visits

Fraud Indicators:

- Appears only during disruption
- No historical linkage to outlet

### 5. Peer Group Validation

Workers are evaluated relative to others in the same zone/outlet.

**Checks:**

- Activity distribution across workers
- Recovery patterns

Fraud Indicators:

- One worker inactive while others are active
- Isolated claims of disruption

### 6. Multi-Signal Location Verification

We combine multiple signals with different trust levels

| Signal Type | Trust Level |
| --- | --- |
| Order data | High |
| Telecom tower | High |
| IP location | Medium |
| GPS | Low |

### 7. Delayed Payout Validation

To prevent exploitation:

- Immediate payout: 5**0%**
- Remaining: **held for audit**

Audit includes:

- Behavioral re-evaluation
- Cluster fraud detection

### 8. Graph-Based Fraud Detection

We construct a **fraud graph**:

**Nodes:**

- Workers
- Devices
- IPs
- Locations

**Edges:**

- Shared IPs
- Same-time activity
- Identical movement patterns

Detect:

- Coordinated spoofing clusters
- Multi-account fraud rings

## Fraud Risk Scoring

Each worker is assigned a dynamic **Fraud Risk Score**:

Risk Score is based on:

- GPS-IP mismatch
- Lack of order activity
- Unrealistic movement
- Peer inconsistency
- Behavioral anomalies
- Outlet mismatch

### Decision Thresholds

| Score Range | Action |
| --- | --- |
| < 30 | Approve |
| 30–60 | Reduce payout |
| 60–80 | Hold for review |
| > 80 | Reject + penalize |

## Key Design Rule

> Location alone is never sufficient for claim approval.
> 

A valid claim must satisfy:

- Verified presence
- Verified work activity
- Verified income impact

## System Advantage

Unlike traditional systems, we:

- Do NOT rely on self-reported GPS
- Use **real economic activity as ground truth**
- Validate claims using **peer-level intelligence**
- Detect fraud using **multi-signal + behavioral analysis**

## Outcome

This approach ensures:

- High fraud resistance
- Fair claim distribution
- Scalable autonomous validation
- Strong trust for partners and workers