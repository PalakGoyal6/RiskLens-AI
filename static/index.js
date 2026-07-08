// Global State Variables
let applicantsList = []; // Caches basic details of 1000 applicants
let currentCohortFilter = "all";
let currentApplicantExplanation = null; // Store active applicant's explanation
let activeApplicantId = null;
let X_test_mean = {}; // Store baseline means if needed
let performanceThreshold = 0.15; // Selected threshold for metric simulations
let rocPoints = []; // Precalculated ROC coordinates

// Feature configurations for mapping back and forth in the input form
const FORM_FEATURES = [
    "EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1",
    "BUREAU_DEBT_CREDIT_RATIO", "AMT_GOODS_PRICE", "AMT_CREDIT",
    "AMT_ANNUITY", "CREDIT_INCOME_RATIO", "CREDIT_TERM",
    "DAYS_EMPLOYED", "LATE_PAYMENT_RATIO", "POS_COUNT",
    "RECENT_LATE_RATIO", "CODE_GENDER_M"
];

// Display Names Dictionary
const FEATURE_DISPLAY_NAMES = {
    "EXT_SOURCE_2": "Credit Bureau Score A",
    "EXT_SOURCE_3": "Credit Bureau Score B",
    "EXT_SOURCE_1": "Credit Bureau Score C",
    "DAYS_EMPLOYED": "Employment Duration",
    "AMT_GOODS_PRICE": "Purchase Price",
    "BUREAU_DEBT_CREDIT_RATIO": "Debt-to-Credit Ratio",
    "PREV_CREDIT_APP_RATIO": "Credit-to-Application Ratio",
    "POS_COUNT": "No. of Previous POS Loans",
    "RECENT_LATE_RATIO": "Recent Late Payment Rate",
    "CREDIT_TERM": "Credit Term Duration",
    "AMT_ANNUITY": "Annual Loan Annuity",
    "AMT_CREDIT": "Total Credit Amount",
    "LATE_PAYMENT_RATIO": "Late Payment History Ratio",
    "CODE_GENDER_M": "Gender: Male",
    "CODE_GENDER_F": "Gender: Female",
    "NAME_FAMILY_STATUS_Married": "Family Status: Married",
    "FLAG_DOCUMENT_3": "Document 3 Submitted",
    "AMT_INCOME_TOTAL": "Total Annual Income",
    "CNT_CHILDREN": "Number of Children",
    "OWN_CAR_AGE": "Age of Applicant's Car",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio",
    "CC_UTILIZATION_TREND_6M": "Credit Card Utilization Trend (6M)",
    "CC_MAX_UTILIZATION": "Credit Card Max Utilization",
    "CC_AVG_UTILIZATION": "Credit Card Avg Utilization",
    "CC_OVER_LIMIT_RATIO": "Credit Card Over-Limit Ratio",
    "CC_DRAWINGS_DIFF_6M": "Credit Card Draw Difference (6M)",
    "PREV_REFUSED_RATIO_RECENT3": "Recent Prev Loan Refusal Rate (Last 3)",
    "PREV_REFUSED_RATIO": "Previous Loan Refusal Rate",
    "PREV_REFUSED_COUNT": "Previous Refused Loan Count",
    "PREV_DAYS_SINCE_LAST_REFUSED": "Days Since Last Loan Refusal",
    "BUREAU_LOAN_COUNT": "Credit Bureau Active Loan Count",
    "BUREAU_ACTIVE_COUNT": "Credit Bureau Active Loans",
    "BUREAU_ACTIVE_CLOSED_RATIO": "Credit Bureau Active-to-Closed Ratio",
    "BUREAU_AVG_DPD_RATIO": "Credit Bureau Average DPD Ratio",
    "BUREAU_DPD_TREND": "Credit Bureau Delinquency Trend",
    "BUREAU_AVG_DEBT": "Credit Bureau Average Debt",
    "BUREAU_AVG_CREDIT": "Credit Bureau Average Credit Limit",
    "INST_LATE_RATIO_3M": "Installment Late Rate (3M)",
    "INST_LATE_RATIO_6M": "Installment Late Rate (6M)",
    "INST_LATE_RATIO_12M": "Installment Late Rate (12M)",
    "MAX_PAYMENT_DELAY": "Max Installment Delay (Days)",
    "AVG_PAYMENT_DELAY": "Average Installment Delay (Days)",
    "RECENT_AVG_DELAY": "Recent Average Payment Delay",
    "UNDERPAID_RATIO": "Underpayment Ratio",
    "ANNUITY_INCOME_RATIO": "Annuity-to-Income Ratio",
    "POS_DPD_RATIO": "POS DPD History Ratio",
    "POS_DPD_TREND": "POS Overdue Payment Trend",
    "POS_DPD_SLOPE": "POS Overdue Slope",
    "POS_COMPLETION_RATE_DIFF": "POS Completion Rate Difference"
};

// Plain English Descriptions Dictionary
const FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_2": "Normalized score from external credit bureau (0 = high risk, 1 = low risk). This is the single strongest predictor of default in this model.",
    "EXT_SOURCE_3": "Normalized score from external credit bureau (0 = high risk, 1 = low risk). This is the second strongest predictor of default in this model.",
    "EXT_SOURCE_1": "Normalized score from external credit bureau (0 = high risk, 1 = low risk). Highly predictive of default risk.",
    "DAYS_EMPLOYED": "Number of days the applicant has been employed. Shorter employment durations typically indicate higher risk.",
    "AMT_GOODS_PRICE": "The price of the goods for which the loan is given. High values increase the total borrowing amount.",
    "BUREAU_DEBT_CREDIT_RATIO": "Total outstanding debt relative to total active credit limit from the credit bureau. Higher ratio indicates higher risk.",
    "PREV_CREDIT_APP_RATIO": "Ratio of credit amount approved compared to what was initially requested. Lower ratios suggest lending constraints.",
    "POS_COUNT": "Number of previous Point-of-Sale cash loans. High frequency of short-term loans can indicate financial strain.",
    "RECENT_LATE_RATIO": "The percentage of payments delayed in the last few installments. Directly indicates delinquency risk.",
    "CREDIT_TERM": "Estimated duration of the loan term. Longer terms increase exposure to default risk.",
    "AMT_ANNUITY": "Annual loan installment payment. Higher annuities increase the monthly repayment burden on the applicant.",
    "AMT_CREDIT": "Total credit amount requested by the applicant. Larger credit sizes increase exposure to risk.",
    "LATE_PAYMENT_RATIO": "Overall history of late payments on previous loans. High ratio shows consistent repayment delays.",
    "CODE_GENDER_M": "Indicates if the applicant is male. Historically correlates with slightly higher default rates in this dataset.",
    "CODE_GENDER_F": "Indicates if the applicant is female. Historically correlates with lower default rates in this dataset.",
    "NAME_FAMILY_STATUS_Married": "Indicates if the applicant is married. Married applicants tend to show slightly higher stability and lower risk.",
    "FLAG_DOCUMENT_3": "Indicates if the main loan application document (Document 3) was submitted. Missing documentation increases risk flags.",
    "AMT_INCOME_TOTAL": "The total annual income reported by the applicant. Higher income typically lowers default risk.",
    "CNT_CHILDREN": "The number of children in the applicant's household. More dependents can increase financial commitments.",
    "OWN_CAR_AGE": "The age of the applicant's primary car. Older cars can indicate lower assets or higher maintenance costs.",
    "CREDIT_INCOME_RATIO": "Credit-to-Income Ratio. Measures the credit size relative to total annual income.",
    "CC_UTILIZATION_TREND_6M": "The 6-month trend of the credit card balance utilization rate. An upward trend suggests growing reliance on credit lines and higher risk.",
    "CC_MAX_UTILIZATION": "The maximum credit card balance utilization rate observed historically across all statement periods.",
    "CC_AVG_UTILIZATION": "The average credit card balance utilization rate observed historically across all statement periods.",
    "CC_OVER_LIMIT_RATIO": "The frequency of months where the credit card balance exceeded the approved credit limit.",
    "CC_DRAWINGS_DIFF_6M": "The difference in average monthly credit card cash/purchase drawings between the last 6 months and the historical average.",
    "PREV_REFUSED_RATIO_RECENT3": "The percentage of the last 3 previous loan applications that were refused by the bank.",
    "PREV_REFUSED_RATIO": "The overall percentage of previous loan applications that were refused by the bank.",
    "PREV_REFUSED_COUNT": "The total count of previous applications that were refused.",
    "PREV_DAYS_SINCE_LAST_REFUSED": "Number of days elapsed since the client's last loan application refusal.",
    "BUREAU_LOAN_COUNT": "The total number of past loans the client has registered in the external credit bureau.",
    "BUREAU_ACTIVE_COUNT": "The number of currently active loans the client has with other financial institutions.",
    "BUREAU_ACTIVE_CLOSED_RATIO": "The ratio of active loans to closed loans in the credit bureau records.",
    "BUREAU_AVG_DPD_RATIO": "The percentage of months where the bureau reports the client had payments overdue.",
    "BUREAU_DPD_TREND": "The trend/slope of overdue days reported by the credit bureau. Positive slope suggests worsening delinquency.",
    "BUREAU_AVG_DEBT": "The average outstanding debt amount across all bureau-reported credit accounts.",
    "BUREAU_AVG_CREDIT": "The average credit limit of accounts reported in the credit bureau.",
    "INST_LATE_RATIO_3M": "The percentage of loan installments paid late in the last 3 months.",
    "INST_LATE_RATIO_6M": "The percentage of loan installments paid late in the last 6 months.",
    "INST_LATE_RATIO_12M": "The percentage of loan installments paid late in the last 12 months.",
    "MAX_PAYMENT_DELAY": "The maximum number of days a payment was delayed past the due date.",
    "AVG_PAYMENT_DELAY": "The average delay in days across all historical installment payments.",
    "RECENT_AVG_DELAY": "The average delay in days of installment payments in recent months.",
    "UNDERPAID_RATIO": "The ratio of installments paid below the billed amount.",
    "ANNUITY_INCOME_RATIO": "The monthly loan annuity payment divided by the applicant's monthly income.",
    "POS_DPD_RATIO": "The percentage of monthly points-of-sale cash loan records with overdue payments.",
    "POS_DPD_TREND": "The trend in days past due for POS cash accounts. An upward trend indicates worsening risk.",
    "POS_DPD_SLOPE": "The rate of change of days past due over time for Point-of-Sale cash accounts.",
    "POS_COMPLETION_RATE_DIFF": "The difference between the expected and actual completion rate of previous POS loans."
};

// State Variables for Global tab charts
let globalImportance = [];
let beeswarmData = {};
let globalChartMode = "beeswarm";
let isGlossaryOpen = false;

// Helper utilities for mappings
function getDisplayName(name) {
    return FEATURE_DISPLAY_NAMES[name] || name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function getDescription(name) {
    return FEATURE_DESCRIPTIONS[name] || `SHAP explanation factor for ${getDisplayName(name)}.`;
}

let sessionToken = localStorage.getItem("token") || null;
let currentPredictionId = null;

async function authorizedFetch(url, options = {}) {
    const token = localStorage.getItem("token");
    if (token) {
        options.headers = options.headers || {};
        options.headers["Authorization"] = `Bearer ${token}`;
    }
    const response = await fetch(url, options);
    if (response.status === 401) {
        logoutUser();
        throw new Error("Session expired. Please log in again.");
    }
    return response;
}

async function initDashboard() {
    initTheme();
    document.getElementById("login-overlay").style.display = "none";
    
    const userStr = localStorage.getItem("user");
    if (userStr) {
        const user = JSON.parse(userStr);
        document.getElementById("sidebar-user").style.display = "flex";
        document.getElementById("sidebar-user-name").textContent = user.username;
        document.getElementById("sidebar-user-avatar").textContent = user.username.substring(0, 1).toUpperCase();
        document.getElementById("sidebar-user-role").textContent = user.role === "admin" ? "Administrator" : "Loan Officer";
    }

    // Fetch summary metrics
    try {
        const summaryReq = await authorizedFetch("/api/summary");
        const summary = await summaryReq.json();
        document.getElementById("header-total-cohort").textContent = summary.total_applicants.toLocaleString();
    } catch (e) {
        console.error("Failed to load header summary metrics:", e);
    }

    await fetchCohortData();
    calculateROCCurveCoordinates();
    renderCohortTable();
    populatePresetsDropdown();
    
    if (applicantsList.length > 0) {
        await loadFormPreset(applicantsList[0].id);
    }
    
    await loadGlobalMetrics();
    initGlossary();
    drawROCCurve();
    updatePerformanceMetrics(performanceThreshold);
    await fetchOverrideStats();
    await loadDashboardHome();
}

// DOMContentLoaded Initialization
document.addEventListener("DOMContentLoaded", async () => {
    // Check if demo query parameter is set to bypass authentication
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get("demo") === "true") {
        try {
            console.log("Bypassing credentials: Fetching guest/demo login token...");
            const demoResp = await fetch("/api/auth/demo", { method: "POST" });
            if (demoResp.ok) {
                const res = await demoResp.json();
                if (res.token) {
                    localStorage.setItem("token", res.token);
                    localStorage.setItem("user", JSON.stringify(res.user));
                    sessionToken = res.token;
                    // Clean URL query params so reload acts normally
                    window.history.replaceState({}, document.title, window.location.pathname);
                }
            }
        } catch (e) {
            console.error("Auto demo login failed:", e);
        }
    }

    if (sessionToken) {
        try {
            await initDashboard();
        } catch (e) {
            console.error("Init error, showing login:", e);
            logoutUser();
        }
    } else {
        window.location.href = "/login";
    }
});

// Switch Main Tabs (SPA)
function switchTab(tabId) {
    document.querySelectorAll(".nav-item").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(tab => tab.classList.remove("active"));
    
    const activeBtn = document.getElementById(`tab-btn-${tabId}`);
    if (activeBtn) activeBtn.classList.add("active");
    
    const activeTab = document.getElementById(`tab-${tabId}`);
    if (activeTab) activeTab.classList.add("active");
    
    // Update header Title and Subtitle dynamically
    const headerTitle = document.getElementById("page-title");
    const headerSubtitle = document.getElementById("page-subtitle");
    
    if (tabId === "home") {
        headerTitle.textContent = "Dashboard Home";
        headerSubtitle.textContent = "Overview of today's underwriting activity, stats, and pending reviews.";
        loadDashboardHome();
    } else if (tabId === "cohort") {
        headerTitle.textContent = "Cohort Explorer";
        headerSubtitle.textContent = "Analyze applicant risk metrics, query loan records, and inspect classifications.";
        renderCohortTable();
    } else if (tabId === "scoring") {
        headerTitle.textContent = "Credit Scoring Form";
        headerSubtitle.textContent = "Score custom credit profiles and inspect individual SHAP contributions.";
        if (currentApplicantExplanation) {
            drawPremiumGauge(currentApplicantExplanation.probability);
            renderWaterfallChart(currentApplicantExplanation);
        }
    } else if (tabId === "global") {
        headerTitle.textContent = "Global Explainability";
        headerSubtitle.textContent = "Explore portfolio-wide model characteristics and feature influences.";
        drawGlobalBarChart(globalImportance);
        if (globalChartMode === "beeswarm") {
            drawBeeswarmPlot(beeswarmData);
        } else {
            drawImpactSummaryPlot(beeswarmData);
        }
    } else if (tabId === "performance") {
        headerTitle.textContent = "Model Performance Simulator";
        headerSubtitle.textContent = "Interact with the decision threshold to simulate model error distributions live.";
        drawROCCurve();
        updatePerformanceMetrics(performanceThreshold);
        fetchAndDrawCalibrationCurve();
    }
}

async function loadDashboardHome() {
    try {
        const userStr = localStorage.getItem("user");
        if (userStr) {
            const user = JSON.parse(userStr);
            const nameEl = document.getElementById("home-officer-name");
            if (nameEl) nameEl.textContent = user.username;
        }

        // Fetch home data from backend
        const res = await authorizedFetch("/api/dashboard/home");
        if (!res.ok) throw new Error("Failed to load dashboard home data");
        const data = await res.json();

        // 1. Update Stats Cards
        document.getElementById("home-pending-count").textContent = data.stats.total_pending.toLocaleString();
        document.getElementById("home-approved-count").textContent = data.stats.total_approved.toLocaleString();
        document.getElementById("home-declined-count").textContent = data.stats.total_rejected.toLocaleString();
        document.getElementById("home-avg-risk").textContent = (data.stats.avg_risk_pending * 100).toFixed(1) + "%";
        document.getElementById("pending-badge-count").textContent = data.stats.total_pending + " Cases";

        // 2. Populate Pending Review Cases List
        const pendingList = document.getElementById("home-pending-list");
        pendingList.innerHTML = "";
        if (data.pending_cases.length === 0) {
            pendingList.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 0.82rem; padding: 20px;">No pending cases remaining in cohort!</div>`;
        } else {
            data.pending_cases.forEach(c => {
                const riskClass = c.risk_classification.toLowerCase().replace(" ", "-");
                const item = document.createElement("div");
                item.className = "pending-item";
                item.innerHTML = `
                    <div class="pending-info">
                        <div class="pending-name">${c.name}</div>
                        <div class="pending-meta">Applicant ID: ${c.id}</div>
                    </div>
                    <div class="pending-action">
                        <span class="badge-risk ${riskClass === 'high-risk' ? 'high' : (riskClass === 'moderate-risk' ? 'moderate' : 'low')}">${c.risk_classification}</span>
                        <button class="btn-review-quick" onclick="startReviewFromHome(${c.id})">Start Review</button>
                    </div>
                `;
                pendingList.appendChild(item);
            });
        }

        // 3. Populate Recent Underwriting Activity List
        const recentList = document.getElementById("home-recent-list");
        recentList.innerHTML = "";
        if (data.recent_activity.length === 0) {
            recentList.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 0.82rem; padding: 20px;">No underwriting activity recorded.</div>`;
        } else {
            data.recent_activity.forEach(act => {
                const decisionClass = act.decision.toLowerCase();
                const probPercent = (act.predicted_probability * 100).toFixed(1) + "%";
                
                // Parse timestamp
                let dateStr = "";
                if (act.decision_time) {
                    try {
                        const parsedDate = new Date(act.decision_time.replace(" ", "T"));
                        dateStr = parsedDate.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                    } catch (err) {
                        dateStr = act.decision_time;
                    }
                }
                
                const item = document.createElement("div");
                item.className = "activity-item";
                item.innerHTML = `
                    <div class="activity-header">
                        <span class="activity-title">${act.applicant_name} (ID: ${act.applicant_id})</span>
                        <span class="activity-badge ${decisionClass}">${act.decision}</span>
                    </div>
                    ${act.notes ? `<div class="activity-notes">"${act.notes}"</div>` : ""}
                    <div class="activity-meta">
                        <span>Risk Score: ${probPercent}</span>
                        <span>Officer: ${act.officer_name || 'System'} | ${dateStr}</span>
                    </div>
                `;
                recentList.appendChild(item);
            });
        }
    } catch (e) {
        console.error("Dashboard Home loading failed:", e);
    }
}

// Redirect and load applicant preset in form
async function startReviewFromHome(applicantId) {
    switchTab('scoring');
    await loadFormPreset(applicantId);
}

// Fetch applicants summary list
async function fetchCohortData() {
    try {
        const r = await authorizedFetch("/api/applicants");
        applicantsList = await r.json();
        
        // Dynamically compute summary stats for header
        const total = applicantsList.length;
        const approved = applicantsList.filter(d => d.probability < performanceThreshold).length;
        const rejected = total - approved;
        const defaults = applicantsList.filter(d => d.target === 1).length;
        
        document.getElementById("cohort-approved-count").textContent = approved;
        document.getElementById("cohort-approved-count").nextElementSibling.textContent = `${(approved/total*100).toFixed(1)}% Approval Rate`;
        document.getElementById("cohort-rejected-count").textContent = rejected;
        document.getElementById("cohort-rejected-count").nextElementSibling.textContent = `${(rejected/total*100).toFixed(1)}% Decline Rate`;
        document.getElementById("cohort-default-count").textContent = defaults;
        document.getElementById("cohort-default-count").nextElementSibling.textContent = `${(defaults/total*100).toFixed(1)}% Defaulter Ratio`;
    } catch (e) {
        console.error("Failed to load applicants cohort data:", e);
    }
}

// Sort and Filter Cohort Table
function renderCohortTable() {
    const tableBody = document.getElementById("cohort-table-body");
    const spinner = document.getElementById("cohort-spinner");
    const table = document.getElementById("cohort-table");
    
    spinner.style.display = "none";
    table.style.display = "table";
    tableBody.innerHTML = "";
    
    const query = document.getElementById("cohort-search").value.toLowerCase();
    const sortVal = document.getElementById("cohort-sort").value;
    
    // Apply filters
    let data = [...applicantsList];
    if (currentCohortFilter === "approved") {
        data = data.filter(d => d.probability < performanceThreshold);
    } else if (currentCohortFilter === "rejected") {
        data = data.filter(d => d.probability >= performanceThreshold);
    } else if (currentCohortFilter === "defaults") {
        data = data.filter(d => d.target === 1);
    } else if (currentCohortFilter === "high-risk") {
        data = data.filter(d => d.probability >= 0.15);
    }
    
    // Apply search query
    if (query) {
        data = data.filter(d => d.id.toString().includes(query));
    }
    
    // Apply sorting
    if (sortVal === "prob-desc") {
        data.sort((a, b) => b.probability - a.probability);
    } else if (sortVal === "prob-asc") {
        data.sort((a, b) => a.probability - b.probability);
    } else if (sortVal === "id-asc") {
        data.sort((a, b) => a.id - b.id);
    }
    
    if (data.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No matching applicants found.</td></tr>`;
        return;
    }
    
    data.forEach(d => {
        const isApproved = d.probability < performanceThreshold;
        const decisionText = isApproved ? "Approved" : "Rejected";
        const decisionClass = isApproved ? "badge safe" : "badge risk";
        
        const riskClass = d.probability >= 0.25 ? "High Risk" : (d.probability >= 0.10 ? "Moderate" : "Low Risk");
        const riskBadge = d.probability >= 0.25 ? "badge risk" : (d.probability >= 0.10 ? "badge amber" : "badge safe");
        
        const targetText = d.target === 1 ? "Defaulted" : "Paid";
        const targetClass = d.target === 1 ? "badge risk" : "badge safe";
        
        const row = document.createElement("tr");
        row.innerHTML = `
            <td style="font-weight:600;">${d.id}</td>
            <td style="font-weight:700;color:var(--text-primary);">${(d.probability * 100).toFixed(1)}%</td>
            <td><span class="${riskBadge}">${riskClass}</span></td>
            <td><span class="${decisionClass}">${decisionText}</span></td>
            <td><span class="${targetClass}">${targetText}</span></td>
            <td>
                <button class="action-link" onclick="event.stopPropagation(); selectApplicantFromTable(${d.id})">
                    <i class="fa-solid fa-address-card"></i> Score Profile
                </button>
            </td>
        `;
        row.onclick = () => selectApplicantFromTable(d.id);
        tableBody.appendChild(row);
    });
}

function setCohortFilter(filterType) {
    currentCohortFilter = filterType;
    document.querySelectorAll(".filter-pills .pill-btn").forEach(btn => {
        btn.classList.remove("active");
        if (btn.textContent.toLowerCase() === filterType || 
            (filterType === "defaults" && btn.textContent.includes("Defaults")) ||
            (filterType === "high-risk" && btn.textContent.includes("High Risk"))) {
            btn.classList.add("active");
        }
    });
    renderCohortTable();
}

function filterCohortTable() {
    renderCohortTable();
}

function selectApplicantFromTable(id) {
    document.getElementById("form-preset-select").value = id;
    loadFormPreset(id);
    switchTab("scoring");
}

// Populate Presets dropdown in Scoring Form
function populatePresetsDropdown() {
    const select = document.getElementById("form-preset-select");
    select.innerHTML = "";
    applicantsList.forEach(app => {
        const option = document.createElement("option");
        option.value = app.id;
        option.textContent = `ID: ${app.id} (Risk: ${(app.probability*100).toFixed(0)}% | Actual: ${app.target === 1 ? 'Default' : 'Paid'})`;
        select.appendChild(option);
    });
}

async function loadRandomFormPreset() {
    if (applicantsList.length === 0) return;
    const randomIdx = Math.floor(Math.random() * applicantsList.length);
    const id = applicantsList[randomIdx].id;
    document.getElementById("form-preset-select").value = id;
    await loadFormPreset(id);
}

// Fetch applicant SHAP details & fill form
async function loadFormPreset(id) {
    activeApplicantId = parseInt(id);
    resetChatForApplicant(id);
    try {
        const r = await authorizedFetch(`/api/applicant/${id}`);
        const data = await r.json();
        currentApplicantExplanation = data;
        currentPredictionId = data.prediction_id;
        
        // Show applicant name badge
        const badge = document.getElementById("active-applicant-name");
        if (data.name) {
            badge.textContent = data.name;
            badge.style.display = "inline-block";
        } else {
            badge.style.display = "none";
        }
        
        // Fetch previous decisions/history for timeline
        await fetchDecisionHistory(id);
        
        // Fill form fields dynamically based on features explanation
        const featuresMap = {};
        data.explanation.forEach(exp => {
            featuresMap[exp.feature] = exp.feature_value;
        });
        
        FORM_FEATURES.forEach(feat => {
            const input = document.getElementById(`input-${feat}`);
            if (input) {
                const val = featuresMap[feat];
                input.value = val !== undefined && val !== null ? val : 0;
            }
        });
        
        // Update Gauge & Waterfall
        drawPremiumGauge(data.probability, data.narration);
        renderWaterfallChart(data);
    } catch (e) {
        console.error("Failed to load applicant preset profile:", e);
    }
}

// Scoring Form Tabs switching
function switchFormTab(formTabId) {
    document.querySelectorAll(".form-tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".form-tab-panel").forEach(panel => panel.classList.remove("active"));
    
    // Find clicked button
    const btn = Array.from(document.querySelectorAll(".form-tab-btn")).find(b => b.textContent.toLowerCase().includes(formTabId.substring(0, 3)));
    if (btn) btn.classList.add("active");
    
    const panel = document.getElementById(`form-panel-${formTabId}`);
    if (panel) panel.classList.add("active");
}

// Submit custom scoring form features to predict & explain
async function analyzeCustomApplicant() {
    const payloadFeatures = {};
    FORM_FEATURES.forEach(feat => {
        const input = document.getElementById(`input-${feat}`);
        if (input) {
            payloadFeatures[feat] = parseFloat(input.value);
        }
    });
    
    try {
        document.getElementById("waterfall-spinner").style.display = "flex";
        
        const recalculateReq = await authorizedFetch(`/api/applicant/${activeApplicantId}/recalculate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: payloadFeatures })
        });
        const data = await recalculateReq.json();
        
        currentApplicantExplanation = data;
        currentPredictionId = data.prediction_id;
        
        // Update Gauge and Waterfall with new predictions
        drawPremiumGauge(data.probability, data.narration);
        renderWaterfallChart(data);
        
        // Refresh the audit timeline history
        await fetchDecisionHistory(activeApplicantId);
    } catch (e) {
        console.error("Failed to score custom applicant features:", e);
    }
}

// D3 Premium Half-Circle speedometer gauge
function drawPremiumGauge(prob, narration = null) {
    const svg = d3.select("#scoring-premium-gauge");
    svg.selectAll("*").remove();
    
    const width = 200;
    const height = 120;
    const radius = 90;
    
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${width / 2}, ${height - 15})`);
        
    // Define gauge color zones (arc segments)
    const arcData = [
        { start: -Math.PI / 2, end: -Math.PI / 2 + (0.15 * Math.PI), color: "var(--accent-teal)" },   // Safe Zone (0-15%)
        { start: -Math.PI / 2 + (0.15 * Math.PI), end: -Math.PI / 2 + (0.35 * Math.PI), color: "var(--accent-gold)" },   // Warning Zone (15-35%)
        { start: -Math.PI / 2 + (0.35 * Math.PI), end: Math.PI / 2, color: "var(--accent-crimson)" }    // Decline Zone (35-100%)
    ];
    
    // Draw Arcs
    arcData.forEach(d => {
        const arc = d3.arc()
            .innerRadius(radius - 24)
            .outerRadius(radius)
            .startAngle(d.start)
            .endAngle(d.end);
            
        chartGroup.append("path")
            .attr("d", arc)
            .attr("fill", d.color)
            .attr("class", "gauge-arc-path");
    });
    
    // Reference border circle segment
    const outlineArc = d3.arc()
        .innerRadius(radius - 24)
        .outerRadius(radius)
        .startAngle(-Math.PI / 2)
        .endAngle(Math.PI / 2);
        
    chartGroup.append("path")
        .attr("d", outlineArc)
        .attr("fill", "none")
        .attr("stroke", "rgba(255, 255, 255, 0.08)")
        .attr("stroke-width", "1px");
        
    // Draw indicator ticks
    const tickAngles = [-90, -60, -30, 0, 30, 60, 90];
    tickAngles.forEach(angle => {
        const radians = (angle * Math.PI) / 180;
        const x1 = (radius - 30) * Math.sin(radians);
        const y1 = -(radius - 30) * Math.cos(radians);
        const x2 = (radius - 24) * Math.sin(radians);
        const y2 = -(radius - 24) * Math.cos(radians);
        
        chartGroup.append("line")
            .attr("x1", x1)
            .attr("y1", y1)
            .attr("x2", x2)
            .attr("y2", y2)
            .attr("stroke", "rgba(255,255,255,0.2)")
            .attr("stroke-width", "1.5px");
    });
    
    // Draw center anchor pin
    chartGroup.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", 3.5)
        .attr("fill", "var(--text-primary)");
        
    // Needle angle calculation
    // prob = 0 maps to -90 deg. prob = 1 maps to +90 deg.
    // However, to align with speedometer arcs visually, let's cap needle at 90 deg.
    const cappedProb = Math.min(1.0, Math.max(0.0, prob));
    const targetAngle = (cappedProb * 180) - 90;
    
    // Draw Needle
    const needle = chartGroup.append("line")
        .attr("class", "gauge-needle")
        .attr("x1", 0)
        .attr("y1", 0)
        .attr("x2", 0)
        .attr("y2", -(radius - 32))
        .attr("stroke", "var(--text-primary)")
        .attr("stroke-width", "2px")
        .attr("stroke-linecap", "round")
        .attr("transform", "rotate(-90)"); // Initial state
        
    // Animate Needle rotation
    needle.transition()
        .duration(1000)
        .ease(d3.easeCubicOut)
        .attr("transform", `rotate(${targetAngle})`);
        
    // Update textual risk values
    animateValue("scoring-risk-score", parseFloat(document.getElementById("scoring-risk-score").textContent) / 100, prob, 800, "%");
    
    const ratingEl = document.getElementById("scoring-risk-rating");
    const badgeEl = document.getElementById("scoring-decision-badge");
    const briefingEl = document.getElementById("scoring-decision-text");
    
    const isApproved = prob < performanceThreshold;
    
    if (prob >= 0.25) {
        ratingEl.textContent = "HIGH RISK";
        ratingEl.style.color = "var(--accent-crimson)";
    } else if (prob >= 0.10) {
        ratingEl.textContent = "MODERATE RISK";
        ratingEl.style.color = "var(--accent-gold)";
    } else {
        ratingEl.textContent = "LOW RISK";
        ratingEl.style.color = "var(--accent-teal)";
    }
    
    if (isApproved) {
        badgeEl.textContent = "Approve";
        badgeEl.className = "brief-value badge safe";
    } else {
        badgeEl.textContent = "Decline";
        badgeEl.className = "brief-value badge risk";
    }

    if (narration) {
        briefingEl.innerHTML = `<span style="color:var(--accent-teal);font-weight:600;display:block;margin-bottom:6px;"><i class="fa-solid fa-robot"></i> AI-Assisted Risk Explanation:</span><span style="line-height:1.4;display:inline-block;">${narration}</span>`;
    } else {
        if (isApproved) {
            briefingEl.innerHTML = `<span style="color:var(--accent-teal);font-weight:600;">Recommend Approval.</span> Applicant default risk (${(prob*100).toFixed(1)}%) is below the current decision cutoff of ${(performanceThreshold*100).toFixed(1)}%. Profile shows stable credit bureau values and manageable debt structure.`;
        } else {
            briefingEl.innerHTML = `<span style="color:var(--accent-crimson);font-weight:600;">Recommend Decline.</span> Applicant default probability (${(prob*100).toFixed(1)}%) exceeds lending safety limit. High risk concentration indicators detected.`;
        }
    }
}

// Generate D3 Horizontal Feature Importance Bar Chart
function drawGlobalBarChart(data) {
    const container = d3.select("#global-bar-chart-container");
    const svg = d3.select("#global-bar-chart");
    
    const width = container.node().getBoundingClientRect().width;
    const height = 480;
    svg.attr("width", width).attr("height", height);
    
    const margin = { top: 20, right: 60, bottom: 40, left: 180 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    svg.selectAll("*").remove();
    
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    const barData = data.slice(0, 15);
    
    const yScale = d3.scaleBand()
        .domain(barData.map(d => getDisplayName(d.feature)))
        .range([0, chartHeight])
        .padding(0.2);
        
    const xScale = d3.scaleLinear()
        .domain([0, d3.max(barData, d => d.importance)])
        .range([0, chartWidth]);
        
    // Draw gridlines
    chartGroup.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5).tickSize(-chartHeight).tickFormat(""));
        
    chartGroup.selectAll(".grid line")
        .style("stroke", "var(--chart-axis)")
        .style("stroke-opacity", 0.7);
    chartGroup.selectAll(".grid path").style("stroke-width", 0);
    
    const defs = svg.append("defs");
    const barGrad = defs.append("linearGradient")
        .attr("id", "bar-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");
    barGrad.append("stop").attr("offset", "0%").attr("stop-color", "var(--accent-indigo)").attr("stop-opacity", 0.6);
    barGrad.append("stop").attr("offset", "100%").attr("stop-color", "var(--accent-blue)").attr("stop-opacity", 0.9);
    
    // Draw Bars
    chartGroup.selectAll(".bar")
        .data(barData)
        .enter()
        .append("rect")
        .attr("class", "bar-chart-rect")
        .attr("y", d => yScale(getDisplayName(d.feature)))
        .attr("x", 0)
        .attr("height", yScale.bandwidth())
        .attr("width", 0)
        .attr("fill", "url(#bar-gradient)")
        .attr("rx", 4)
        .on("mouseover", function(event, d) {
            showTooltip(event, `
                <strong>${getDisplayName(d.feature)}</strong><br>
                Average Magnitude of Impact: <strong>${d.importance.toFixed(4)}</strong><br>
                <span style="color:var(--text-secondary); font-size:0.75rem;">${getDescription(d.feature)}</span>
            `);
        })
        .on("mouseout", hideTooltip)
        .transition()
        .duration(800)
        .delay((d, i) => i * 30)
        .attr("width", d => xScale(d.importance));
        
    const yAxisGroup = chartGroup.append("g")
        .call(d3.axisLeft(yScale).tickSize(0));
        
    yAxisGroup.selectAll("text")
        .style("font-size", "0.8rem")
        .style("font-weight", "500")
        .style("fill", "var(--text-secondary)")
        .attr("dx", "-8px")
        .style("cursor", "help")
        .on("mouseover", function(event, d) {
            const item = barData.find(b => getDisplayName(b.feature) === d);
            const rawKey = item ? item.feature : "";
            showTooltip(event, `<strong>${d}</strong><br>${getDescription(rawKey)}`);
        })
        .on("mouseout", hideTooltip);
        
    chartGroup.selectAll(".val-label")
        .data(barData)
        .enter()
        .append("text")
        .attr("class", "val-label")
        .attr("x", d => xScale(d.importance) + 8)
        .attr("y", d => yScale(getDisplayName(d.feature)) + yScale.bandwidth() / 2 + 4)
        .style("fill", "var(--text-primary)")
        .style("font-weight", "600")
        .text(d => d.importance.toFixed(3))
        .style("opacity", 0)
        .transition()
        .duration(800)
        .delay(400)
        .style("opacity", 1);
        
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 35)
        .attr("text-anchor", "middle")
        .text("Mean Absolute SHAP Value (Average Magnitude of Risk Contribution)");
        
    svg.selectAll(".domain").style("stroke", "var(--chart-axis)");
}

// Fetch Global importance and beeswarm data
async function loadGlobalMetrics() {
    try {
        const r1 = await authorizedFetch("/api/global-importance");
        globalImportance = await r1.json();
        
        const r2 = await authorizedFetch("/api/beeswarm");
        beeswarmData = await r2.json();
    } catch (e) {
        console.error("Failed to load global SHAP details:", e);
    }
}

// Toggle global chart layout modes
function setGlobalChartMode(mode) {
    globalChartMode = mode;
    document.querySelectorAll(".segmented-control .control-btn").forEach(btn => btn.classList.remove("active"));
    
    if (mode === "beeswarm") {
        document.getElementById("btn-chart-beeswarm").classList.add("active");
        document.getElementById("beeswarm-plot").style.display = "block";
        document.getElementById("impact-summary-plot").style.display = "none";
        document.getElementById("beeswarm-legend").style.display = "block";
        document.getElementById("beeswarm-title").textContent = "SHAP Beeswarm Plot";
        document.getElementById("beeswarm-description").textContent = "Displays the distribution of impact that each of the top 15 features has on all predictions. Points represent individual applicants.";
        drawBeeswarmPlot(beeswarmData);
    } else {
        document.getElementById("btn-chart-summary").classList.add("active");
        document.getElementById("beeswarm-plot").style.display = "none";
        document.getElementById("impact-summary-plot").style.display = "block";
        document.getElementById("beeswarm-legend").style.display = "none";
        document.getElementById("beeswarm-title").textContent = "Risk Impact Summary";
        document.getElementById("beeswarm-description").textContent = "The percentage of applicants in the portfolio where this factor increased or decreased default risk.";
        drawImpactSummaryPlot(beeswarmData);
    }
}

// Draw D3 Beeswarm plot
function drawBeeswarmPlot(data) {
    const container = d3.select("#beeswarm-container");
    const svg = d3.select("#beeswarm-plot");
    
    const width = container.node().getBoundingClientRect().width;
    const height = 560;
    svg.attr("width", width).attr("height", height);
    
    const margin = { top: 20, right: 40, bottom: 45, left: 185 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    svg.selectAll("*").remove();
    
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    const features = Object.keys(data);
    
    const yScale = d3.scaleBand()
        .domain(features.map(f => getDisplayName(f)))
        .range([0, chartHeight])
        .padding(0.5);
        
    let maxShap = 0;
    features.forEach(feat => {
        data[feat].forEach(pt => {
            if (Math.abs(pt.shap_value) > maxShap) {
                maxShap = Math.abs(pt.shap_value);
            }
        });
    });
    
    const xScale = d3.scaleLinear()
        .domain([-maxShap * 1.05, maxShap * 1.05])
        .range([0, chartWidth]);
        
    const colorScale = d3.scaleSequential()
        .domain([0, 1])
        .interpolator(d3.interpolateRgbBasis(["#38bdf8", "#818cf8", "#f43f5e"]));
        
    chartGroup.append("line")
        .attr("x1", xScale(0))
        .attr("y1", 0)
        .attr("x2", xScale(0))
        .attr("y2", chartHeight)
        .style("stroke", "#475569")
        .style("stroke-dasharray", "4, 4")
        .style("stroke-width", "1.5px");
        
    const nodes = [];
    features.forEach(feat => {
        data[feat].forEach(pt => {
            nodes.push({
                x: xScale(pt.shap_value),
                y: yScale(getDisplayName(feat)) + yScale.bandwidth() / 2,
                targetY: yScale(getDisplayName(feat)) + yScale.bandwidth() / 2,
                targetX: xScale(pt.shap_value),
                shap_value: pt.shap_value,
                feature_value: pt.feature_value,
                normalized_value: pt.normalized_value,
                applicant_id: pt.applicant_id,
                feature: feat
            });
        });
    });
    
    const simulation = d3.forceSimulation(nodes)
        .force("x", d3.forceX(d => d.targetX).strength(1.5))
        .force("y", d3.forceY(d => d.targetY).strength(0.8))
        .force("collide", d3.forceCollide(2.8))
        .stop();
        
    for (let i = 0; i < 120; ++i) simulation.tick();
    
    features.forEach(feat => {
        chartGroup.append("line")
            .attr("x1", 0)
            .attr("y1", yScale(getDisplayName(feat)) + yScale.bandwidth() / 2)
            .attr("x2", chartWidth)
            .attr("y2", yScale(getDisplayName(feat)) + yScale.bandwidth() / 2)
            .style("stroke", "var(--chart-axis)")
            .style("stroke-width", "1px");
    });
    
    chartGroup.selectAll(".beeswarm-node")
        .data(nodes)
        .enter()
        .append("circle")
        .attr("class", "beeswarm-node")
        .attr("cx", d => d.x)
        .attr("cy", d => d.y)
        .attr("r", 2.2)
        .style("fill", d => colorScale(d.normalized_value))
        .style("fill-opacity", 0.8)
        .on("mouseover", function(event, d) {
            const rawValText = d.feature_value !== null ? d.feature_value.toFixed(4) : "N/A";
            showTooltip(event, `
                <strong>Applicant ID:</strong> ${d.applicant_id}<br>
                <strong>Factor:</strong> ${getDisplayName(d.feature)}<br>
                <strong>Value:</strong> ${rawValText}<br>
                <strong>Risk Impact:</strong> <strong style="color:${d.shap_value >= 0 ? 'var(--accent-crimson)' : 'var(--accent-teal)'}">${d.shap_value >= 0 ? '+' : ''}${d.shap_value.toFixed(4)} logits</strong>
            `);
            d3.select(this).attr("r", 5).style("fill-opacity", 1);
        })
        .on("mouseout", function() {
            hideTooltip();
            d3.select(this).attr("r", 2.2).style("fill-opacity", 0.8);
        })
        .on("click", function(event, d) {
            selectApplicantFromTable(d.applicant_id);
        });
        
    const yAxis = chartGroup.append("g")
        .call(d3.axisLeft(yScale).tickSize(0));
        
    yAxis.selectAll("text")
        .style("font-size", "0.78rem")
        .style("font-weight", "500")
        .style("fill", "var(--text-secondary)")
        .attr("dx", "-8px")
        .style("cursor", "help")
        .on("mouseover", function(event, d) {
            const featKey = features.find(f => getDisplayName(f) === d);
            showTooltip(event, `<strong>${d}</strong><br>${getDescription(featKey)}`);
        })
        .on("mouseout", hideTooltip);
        
    const xAxisGroup = chartGroup.append("g")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5));
        
    xAxisGroup.selectAll("text")
        .style("font-size", "0.75rem")
        .style("fill", "var(--text-secondary)");
        
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 35)
        .attr("text-anchor", "middle")
        .text("SHAP Value (Risk Log-Odds Impact: Left = Good, Right = Bad)");
        
    svg.selectAll(".domain").style("stroke", "var(--chart-axis)");
    svg.selectAll(".tick line").style("stroke", "var(--chart-axis)");
}

// Draw D3 Global Risk impact summary
function drawImpactSummaryPlot(data) {
    const container = d3.select("#beeswarm-container");
    const svg = d3.select("#impact-summary-plot");
    
    const width = container.node().getBoundingClientRect().width;
    const height = 560;
    svg.attr("width", width).attr("height", height);
    
    const margin = { top: 30, right: 60, bottom: 45, left: 185 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    svg.selectAll("*").remove();
    
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    const features = Object.keys(data);
    
    const summaryData = features.map(feat => {
        const pts = data[feat];
        const increased = pts.filter(p => p.shap_value > 0.0001).length;
        const decreased = pts.filter(p => p.shap_value < -0.0001).length;
        const total = pts.length;
        return {
            feature: feat,
            displayName: getDisplayName(feat),
            increasedPct: (increased / total) * 100,
            decreasedPct: (decreased / total) * 100,
            neutralPct: ((total - increased - decreased) / total) * 100
        };
    });
    
    const yScale = d3.scaleBand()
        .domain(summaryData.map(d => d.displayName))
        .range([0, chartHeight])
        .padding(0.35);
        
    const xScale = d3.scaleLinear()
        .domain([-100, 100])
        .range([0, chartWidth]);
        
    chartGroup.append("line")
        .attr("x1", xScale(0))
        .attr("y1", 0)
        .attr("x2", xScale(0))
        .attr("y2", chartHeight)
        .style("stroke", "#475569")
        .style("stroke-width", "1.5px");
        
    chartGroup.append("g")
        .attr("class", "grid")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => Math.abs(d) + "%").tickSize(-chartHeight));
        
    chartGroup.selectAll(".grid line")
        .style("stroke", "var(--chart-axis)")
        .style("stroke-opacity", 0.5);
    chartGroup.selectAll(".grid path").style("stroke-width", 0);
    
    const rows = chartGroup.selectAll(".summary-row")
        .data(summaryData)
        .enter()
        .append("g")
        .attr("class", "summary-row");
        
    rows.append("rect")
        .attr("y", d => yScale(d.displayName))
        .attr("x", xScale(0))
        .attr("height", yScale.bandwidth())
        .attr("width", 0)
        .style("fill", "var(--accent-teal)")
        .style("fill-opacity", 0.85)
        .attr("rx", 3)
        .on("mouseover", function(event, d) {
            showTooltip(event, `
                <strong>${d.displayName}</strong><br>
                Decreased Default Risk for: <strong>${d.decreasedPct.toFixed(1)}%</strong> of applicants.<br>
                <span style="color:var(--text-secondary); font-size:0.75rem;">${getDescription(d.feature)}</span>
            `);
        })
        .on("mouseout", hideTooltip)
        .transition()
        .duration(800)
        .attr("x", d => xScale(-d.decreasedPct))
        .attr("width", d => xScale(0) - xScale(-d.decreasedPct));
        
    rows.append("rect")
        .attr("y", d => yScale(d.displayName))
        .attr("x", xScale(0))
        .attr("height", yScale.bandwidth())
        .attr("width", 0)
        .style("fill", "var(--accent-crimson)")
        .style("fill-opacity", 0.85)
        .attr("rx", 3)
        .on("mouseover", function(event, d) {
            showTooltip(event, `
                <strong>${d.displayName}</strong><br>
                Increased Default Risk for: <strong>${d.increasedPct.toFixed(1)}%</strong> of applicants.<br>
                <span style="color:var(--text-secondary); font-size:0.75rem;">${getDescription(d.feature)}</span>
            `);
        })
        .on("mouseout", hideTooltip)
        .transition()
        .duration(800)
        .attr("width", d => xScale(d.increasedPct) - xScale(0));
        
    rows.append("text")
        .attr("x", d => xScale(-d.decreasedPct) - 6)
        .attr("y", d => yScale(d.displayName) + yScale.bandwidth() / 2 + 4)
        .attr("text-anchor", "end")
        .style("fill", "var(--accent-teal)")
        .style("font-size", "0.72rem")
        .style("font-weight", "600")
        .text(d => d.decreasedPct > 5 ? `${d.decreasedPct.toFixed(0)}%` : "");
        
    rows.append("text")
        .attr("x", d => xScale(d.increasedPct) + 6)
        .attr("y", d => yScale(d.displayName) + yScale.bandwidth() / 2 + 4)
        .attr("text-anchor", "start")
        .style("fill", "var(--accent-crimson)")
        .style("font-size", "0.72rem")
        .style("font-weight", "600")
        .text(d => d.increasedPct > 5 ? `${d.increasedPct.toFixed(0)}%` : "");
        
    const yAxis = chartGroup.append("g")
        .call(d3.axisLeft(yScale).tickSize(0));
        
    yAxis.selectAll("text")
        .style("font-size", "0.78rem")
        .style("font-weight", "500")
        .style("fill", "var(--text-secondary)")
        .attr("dx", "-8px")
        .style("cursor", "help")
        .on("mouseover", function(event, d) {
            const item = summaryData.find(s => s.displayName === d);
            const featKey = item ? item.feature : "";
            showTooltip(event, `<strong>${d}</strong><br>${getDescription(featKey)}`);
        })
        .on("mouseout", hideTooltip);
        
    const xAxisGroup = chartGroup.append("g")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => Math.abs(d) + "%"));
        
    xAxisGroup.selectAll("text")
        .style("font-size", "0.75rem")
        .style("fill", "var(--text-secondary)");
        
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 35)
        .attr("text-anchor", "middle")
        .text("Risk Impact Rate (% of Applicants: Left = Risk Decreased, Right = Risk Increased)");
        
    svg.selectAll(".domain").style("stroke", "var(--chart-axis)");
    svg.selectAll(".tick line").style("stroke", "var(--chart-axis)");
}

// Generate local Waterfall Chart using D3
function renderWaterfallChart(data) {
    const container = d3.select("#waterfall-container");
    const svg = d3.select("#waterfall-chart");
    
    const width = container.node().getBoundingClientRect().width;
    const height = 450;
    svg.attr("width", width).attr("height", height);
    
    const margin = { top: 30, right: 80, bottom: 45, left: 195 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    svg.selectAll("*").remove();
    
    document.getElementById("waterfall-spinner").style.display = "none";
    
    const rawExplanation = data.explanation;
    const topCount = 6; // Compact waterfall
    const mainFeatures = rawExplanation.slice(0, topCount);
    
    let otherSum = 0;
    rawExplanation.slice(topCount).forEach(item => {
        otherSum += item.shap_value;
    });
    
    const items = [];
    items.push({
        label: "Baseline (Average Risk)",
        featureName: "BASE_VALUE",
        val: data.base_value,
        type: "base"
    });
    
    mainFeatures.forEach(item => {
        const dName = getDisplayName(item.feature);
        let rawText = "N/A";
        if (item.feature_value !== null) {
            if (item.feature.includes("RATIO") || item.feature.includes("RATE")) {
                rawText = (item.feature_value * 100).toFixed(0) + "%";
            } else if (item.feature.includes("AMT") || item.feature.includes("PRICE") || item.feature.includes("ANNUITY")) {
                rawText = "₹" + Math.round(item.feature_value).toLocaleString();
            } else if (item.feature.includes("DAYS_EMPLOYED")) {
                rawText = (Math.abs(item.feature_value) / 365.25).toFixed(1) + " yrs";
            } else {
                rawText = item.feature_value.toFixed(2);
            }
        }
        items.push({
            label: `${dName}: ${rawText}`,
            featureName: item.feature,
            val: item.shap_value,
            type: "contrib"
        });
    });
    
    if (otherSum !== 0) {
        items.push({
            label: "Other 180+ Factors",
            featureName: "OTHER_FEATURES",
            val: otherSum,
            type: "contrib"
        });
    }
    
    let runningSum = data.base_value;
    const steps = [];
    
    steps.push({
        label: items[0].label,
        featureName: items[0].featureName,
        start: 0,
        end: items[0].val,
        value: items[0].val,
        type: "base"
    });
    
    runningSum = items[0].val;
    
    for (let i = 1; i < items.length; i++) {
        const item = items[i];
        steps.push({
            label: item.label,
            featureName: item.featureName,
            start: runningSum,
            end: runningSum + item.val,
            value: item.val,
            type: "contrib"
        });
        runningSum += item.val;
    }
    
    steps.push({
        label: "Final Score Prediction",
        featureName: "FINAL_PREDICTION",
        start: 0,
        end: runningSum,
        value: runningSum,
        type: "final"
    });
    
    steps.forEach((d, i) => {
        if (d.type === "base") {
            d.prob_start = 0;
            d.prob_end = 1 / (1 + Math.exp(-d.end));
            d.prob_diff = d.prob_end;
            d.labelText = `${(d.prob_diff * 100).toFixed(1)}%`;
        } else if (d.type === "final") {
            d.prob_start = 0;
            d.prob_end = 1 / (1 + Math.exp(-d.end));
            d.prob_diff = d.prob_end;
            d.labelText = `${(d.prob_diff * 100).toFixed(1)}%`;
        } else {
            d.prob_start = 1 / (1 + Math.exp(-d.start));
            d.prob_end = 1 / (1 + Math.exp(-d.end));
            d.prob_diff = d.prob_end - d.prob_start;
            const sign = d.prob_diff >= 0 ? "+" : "";
            const arrow = d.prob_diff >= 0 ? "↑" : "↓";
            d.labelText = `${sign}${(d.prob_diff * 100).toFixed(1)}% ${arrow}`;
        }
    });
    
    const yScale = d3.scaleBand()
        .domain(steps.map((d, i) => i))
        .range([0, chartHeight])
        .padding(0.25);
        
    let minX = 0;
    let maxX = 0;
    steps.forEach(d => {
        const lower = Math.min(d.start, d.end);
        const upper = Math.max(d.start, d.end);
        if (lower < minX) minX = lower;
        if (upper > maxX) maxX = upper;
    });
    
    minX = minX - Math.abs(minX * 0.1) - 0.2;
    maxX = maxX + Math.abs(maxX * 0.1) + 0.2;
    
    const xScale = d3.scaleLinear()
        .domain([minX, maxX])
        .range([0, chartWidth]);
        
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    chartGroup.append("line")
        .attr("x1", xScale(0))
        .attr("y1", 0)
        .attr("x2", xScale(0))
        .attr("y2", chartHeight)
        .style("stroke", "#334155")
        .style("stroke-width", "1px");
        
    for (let i = 0; i < steps.length - 1; i++) {
        const current = steps[i];
        const connectX = current.end;
        const yStart = yScale(i) + yScale.bandwidth() / 2;
        const yEnd = yScale(i + 1) + yScale.bandwidth() / 2;
        
        chartGroup.append("line")
            .attr("class", "waterfall-connector")
            .attr("x1", xScale(connectX))
            .attr("y1", yStart)
            .attr("x2", xScale(connectX))
            .attr("y2", yEnd);
    }
    
    const bar = chartGroup.selectAll(".waterfall-bar")
        .data(steps)
        .enter()
        .append("g")
        .attr("class", "waterfall-bar");
        
    bar.append("rect")
        .attr("y", (d, i) => yScale(i))
        .attr("x", d => xScale(Math.min(d.start, d.end)))
        .attr("height", yScale.bandwidth())
        .attr("width", d => Math.abs(xScale(d.end) - xScale(d.start)))
        .attr("rx", 3)
        .style("fill", d => {
            if (d.type === "base") return "var(--accent-indigo)";
            if (d.type === "final") return "url(#final-grad)";
            return d.value >= 0 ? "var(--accent-crimson)" : "var(--accent-teal)";
        })
        .on("mouseover", function(event, d) {
            let tooltipHtml = "";
            if (d.type === "base") {
                tooltipHtml = `
                    <strong>Average Risk Baseline</strong><br>
                    Log-Odds: <strong>${d.value.toFixed(4)}</strong><br>
                    Base Probability: <strong>${(d.prob_end * 100).toFixed(2)}%</strong>
                `;
            } else if (d.type === "final") {
                const isHighRisk = d.prob_end >= performanceThreshold;
                const statusColor = isHighRisk ? 'var(--accent-crimson)' : 'var(--accent-teal)';
                tooltipHtml = `
                    <strong>Final Credit Risk Prediction</strong><br>
                    Predicted Probability: <strong style="color:${statusColor}">${(d.prob_end * 100).toFixed(2)}%</strong><br>
                    Decision Cutoff: <strong>${(performanceThreshold*100).toFixed(2)}%</strong><br>
                    Log-Odds: <strong>${d.value.toFixed(4)}</strong>
                `;
            } else {
                const directionText = d.prob_diff >= 0 ? "Increases risk ↑" : "Decreases risk ↓";
                const sign = d.prob_diff >= 0 ? "+" : "";
                const statusColor = d.prob_diff >= 0 ? 'var(--accent-crimson)' : 'var(--accent-teal)';
                tooltipHtml = `
                    <strong>${d.label}</strong><br>
                    Risk Shift: <strong style="color:${statusColor}">${sign}${(d.prob_diff * 100).toFixed(2)}% (${directionText})</strong><br>
                    SHAP Value: <strong>${d.value >= 0 ? '+' : ''}${d.value.toFixed(4)} logits</strong><br>
                    <span style="color:var(--text-secondary); font-size:0.75rem;">${getDescription(d.featureName)}</span>
                `;
            }
            showTooltip(event, tooltipHtml);
        })
        .on("mouseout", hideTooltip);
        
    bar.append("text")
        .attr("x", d => {
            const rightEdge = xScale(Math.max(d.start, d.end));
            const leftEdge = xScale(Math.min(d.start, d.end));
            return d.value >= 0 ? rightEdge + 8 : leftEdge - 8;
        })
        .attr("y", (d, i) => yScale(i) + yScale.bandwidth() / 2 + 4)
        .attr("text-anchor", d => d.value >= 0 ? "start" : "end")
        .style("fill", d => {
            if (d.type === "base" || d.type === "final") return "var(--text-primary)";
            return d.value >= 0 ? "var(--accent-crimson)" : "var(--accent-teal)";
        })
        .style("font-weight", "600")
        .text(d => d.labelText);
        
    const yAxisGroup = chartGroup.append("g")
        .call(d3.axisLeft(yScale).tickSize(0).tickFormat((d, i) => steps[i].label));
        
    yAxisGroup.selectAll("text")
        .style("font-size", "0.76rem")
        .style("font-weight", "500")
        .style("fill", "var(--text-secondary)")
        .attr("dx", "-8px")
        .style("cursor", "help")
        .on("mouseover", function(event, d) {
            const idx = parseInt(d);
            const feat = steps[idx];
            const featName = feat ? feat.featureName : "";
            const displayName = feat ? feat.label : "";
            showTooltip(event, `<strong>${displayName}</strong><br>${getDescription(featName)}`);
        })
        .on("mouseout", hideTooltip);
        
    const xAxisGroup = chartGroup.append("g")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5));
        
    xAxisGroup.selectAll("text")
        .style("font-size", "0.75rem")
        .style("fill", "var(--text-secondary)");
        
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 35)
        .attr("text-anchor", "middle")
        .text("Risk Score (Log-Odds Contribution Space)");
        
    const defs = svg.append("defs");
    const finalGrad = defs.append("linearGradient")
        .attr("id", "final-grad")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");
    
    const finalProb = 1 / (1 + Math.exp(-runningSum));
    if (finalProb >= performanceThreshold) {
        finalGrad.append("stop").attr("offset", "0%").attr("stop-color", "var(--accent-indigo)");
        finalGrad.append("stop").attr("offset", "100%").attr("stop-color", "var(--accent-crimson)");
    } else {
        finalGrad.append("stop").attr("offset", "0%").attr("stop-color", "var(--accent-indigo)");
        finalGrad.append("stop").attr("offset", "100%").attr("stop-color", "var(--accent-teal)");
    }
    
    svg.selectAll(".domain").style("stroke", "var(--chart-axis)");
    svg.selectAll(".tick line").style("stroke", "var(--chart-axis)");
}

// ==========================================================================
// Model Performance & Threshold Simulator Logic
// ==========================================================================

function calculateROCCurveCoordinates() {
    if (applicantsList.length === 0) return;
    
    // Sort applicants by probability descending
    const sorted = [...applicantsList].sort((a, b) => b.probability - a.probability);
    const totalPos = sorted.filter(d => d.target === 1).length;
    const totalNeg = sorted.length - totalPos;
    
    rocPoints = [{ fpr: 0, tpr: 0, threshold: 1.0 }];
    let tp = 0;
    let fp = 0;
    
    for (let i = 0; i < sorted.length; i++) {
        if (sorted[i].target === 1) {
            tp++;
        } else {
            fp++;
        }
        rocPoints.push({
            fpr: fp / (totalNeg || 1),
            tpr: tp / (totalPos || 1),
            threshold: sorted[i].probability
        });
    }
    rocPoints.push({ fpr: 1, tpr: 1, threshold: 0.0 });
}

function updatePerformanceMetrics(valStr) {
    const threshold = parseFloat(valStr);
    performanceThreshold = threshold;
    
    // Update Slider text label
    document.getElementById("slider-threshold-val").textContent = threshold.toFixed(3);
    document.getElementById("header-threshold-val").textContent = threshold.toFixed(3);
    
    // Recalculate Confusion Matrix locally for 1000 items
    let tp = 0, fp = 0, tn = 0, fn = 0;
    applicantsList.forEach(d => {
        const pred = d.probability >= threshold ? 1 : 0;
        const actual = d.target;
        
        if (pred === 1 && actual === 1) tp++;
        else if (pred === 1 && actual === 0) fp++;
        else if (pred === 0 && actual === 0) tn++;
        else if (pred === 0 && actual === 1) fn++;
    });
    
    const total = applicantsList.length || 1000;
    
    // Calculate metric ratios
    const precision = tp + fp > 0 ? (tp / (tp + fp)) : 0;
    const recall = tp + fn > 0 ? (tp / (tp + fn)) : 0;
    const accuracy = (tp + tn) / total;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    
    // Update metric cards
    document.getElementById("perf-accuracy").textContent = `${(accuracy * 100).toFixed(1)}%`;
    document.getElementById("perf-f1").textContent = `${(f1 * 100).toFixed(1)}%`;
    document.getElementById("perf-precision").textContent = `${(precision * 100).toFixed(1)}%`;
    document.getElementById("perf-recall").textContent = `${(recall * 100).toFixed(1)}%`;
    
    // Update Confusion Matrix Grid counts and percentages
    document.getElementById("count-tn").textContent = tn;
    document.getElementById("pct-tn").textContent = `${((tn / total) * 100).toFixed(1)}%`;
    
    document.getElementById("count-fp").textContent = fp;
    document.getElementById("pct-fp").textContent = `${((fp / total) * 100).toFixed(1)}%`;
    
    document.getElementById("count-fn").textContent = fn;
    document.getElementById("pct-fn").textContent = `${((fn / total) * 100).toFixed(1)}%`;
    
    document.getElementById("count-tp").textContent = tp;
    document.getElementById("pct-tp").textContent = `${((tp / total) * 100).toFixed(1)}%`;
    
    // Dynamic opacity shifts for tile intensity
    document.getElementById("cell-tn").style.backgroundColor = `rgba(20, 184, 166, ${0.05 + 0.4 * (tn / total)})`;
    document.getElementById("cell-tp").style.backgroundColor = `rgba(20, 184, 166, ${0.05 + 0.4 * (tp / total)})`;
    document.getElementById("cell-fp").style.backgroundColor = `rgba(244, 63, 94, ${0.05 + 0.4 * (fp / total)})`;
    document.getElementById("cell-fn").style.backgroundColor = `rgba(244, 63, 94, ${0.05 + 0.4 * (fn / total)})`;
    
    // Move glowing tracking dot on D3 ROC Curve
    moveROCCurveTrackingDot(threshold);
}

function drawROCCurve() {
    const container = d3.select("#roc-chart-container");
    const svg = d3.select("#roc-chart");
    
    const width = container.node().getBoundingClientRect().width;
    const height = 380;
    svg.attr("width", width).attr("height", height);
    
    const margin = { top: 20, right: 30, bottom: 45, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    svg.selectAll("*").remove();
    
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, chartWidth]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([chartHeight, 0]);
    
    // Draw grid axes lines
    chartGroup.append("g")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5))
        .selectAll("text").style("fill", "var(--text-secondary)");
        
    chartGroup.append("g")
        .call(d3.axisLeft(yScale).ticks(5))
        .selectAll("text").style("fill", "var(--text-secondary)");
        
    // Draw diagonal baseline (50/50 prediction odds)
    chartGroup.append("line")
        .attr("x1", xScale(0))
        .attr("y1", yScale(0))
        .attr("x2", xScale(1))
        .attr("y2", yScale(1))
        .style("stroke", "#334155")
        .style("stroke-dasharray", "4, 4")
        .style("stroke-width", "1px");
        
    // Draw ROC Line Path
    const line = d3.line()
        .x(d => xScale(d.fpr))
        .y(d => yScale(d.tpr));
        
    chartGroup.append("path")
        .datum(rocPoints)
        .attr("class", "roc-path-line")
        .attr("d", line)
        .style("fill", "none")
        .style("stroke", "var(--accent-blue)")
        .style("stroke-width", "2.5px")
        .style("filter", "drop-shadow(0 0 4px rgba(56, 189, 248, 0.5))");
        
    // Draw Axis Title labels
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 35)
        .attr("text-anchor", "middle")
        .text("False Positive Rate (FPR)");
        
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -chartHeight / 2)
        .attr("y", -35)
        .attr("text-anchor", "middle")
        .text("True Positive Rate (TPR)");
        
    // Add Glowing Dot tracker placeholder
    chartGroup.append("circle")
        .attr("id", "roc-tracking-dot")
        .attr("r", 6)
        .style("fill", "var(--accent-indigo)")
        .style("stroke", "var(--text-primary)")
        .style("stroke-width", "2px")
        .style("filter", "drop-shadow(0 0 8px rgba(99, 102, 241, 0.9))")
        .attr("cx", xScale(0))
        .attr("cy", yScale(0));
        
    svg.selectAll(".domain").style("stroke", "var(--chart-axis)");
    svg.selectAll(".tick line").style("stroke", "var(--chart-axis)");
}

function moveROCCurveTrackingDot(threshold) {
    const dot = d3.select("#roc-tracking-dot");
    if (dot.empty() || rocPoints.length === 0) return;
    
    // Find ROC point closest to selected threshold
    let closestPt = rocPoints[0];
    let minDiff = Math.abs(rocPoints[0].threshold - threshold);
    
    for (let i = 1; i < rocPoints.length; i++) {
        const diff = Math.abs(rocPoints[i].threshold - threshold);
        if (diff < minDiff) {
            minDiff = diff;
            closestPt = rocPoints[i];
        }
    }
    
    // Compute positions based on scales
    const container = d3.select("#roc-chart-container");
    if (container.empty()) return;
    
    const width = container.node().getBoundingClientRect().width;
    const margin = { left: 50, right: 30 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = 380 - 20 - 45;
    
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, chartWidth]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([chartHeight, 0]);
    
    dot.transition()
        .duration(200)
        .attr("cx", xScale(closestPt.fpr))
        .attr("cy", yScale(closestPt.tpr));
}

async function fetchAndDrawCalibrationCurve() {
    try {
        const response = await authorizedFetch("/api/model/calibration");
        const data = await response.json();
        
        // Update Brier Scores in UI
        document.getElementById("calib-brier-before").textContent = data.brier_score_before.toFixed(4);
        document.getElementById("calib-brier-after").textContent = data.brier_score_after.toFixed(4);
        
        drawCalibrationChart(data);
    } catch (e) {
        console.error("Error loading calibration data:", e);
    }
}

function drawCalibrationChart(data) {
    const container = d3.select("#calibration-chart-container");
    const svg = d3.select("#calibration-chart");
    if (svg.empty()) return;
    
    // Scale chart size nicely
    const width = container.node().getBoundingClientRect().width * 0.65 || 500;
    const height = 380;
    svg.attr("width", width).attr("height", height);
    
    const margin = { top: 20, right: 30, bottom: 45, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    svg.selectAll("*").remove();
    
    const chartGroup = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);
        
    const xScale = d3.scaleLinear().domain([0, 1]).range([0, chartWidth]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([chartHeight, 0]);
    
    // Draw grid axes lines
    chartGroup.append("g")
        .attr("transform", `translate(0, ${chartHeight})`)
        .call(d3.axisBottom(xScale).ticks(5).tickFormat(d3.format(".0%")))
        .selectAll("text").style("fill", "var(--text-secondary)");
        
    chartGroup.append("g")
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format(".0%")))
        .selectAll("text").style("fill", "var(--text-secondary)");
        
    // Draw diagonal baseline (Perfect Calibration)
    chartGroup.append("line")
        .attr("x1", xScale(0))
        .attr("y1", yScale(0))
        .attr("x2", xScale(1))
        .attr("y2", yScale(1))
        .style("stroke", "#64748b")
        .style("stroke-dasharray", "4, 4")
        .style("stroke-width", "1.5px");
        
    // Transform arrays into object structures
    const beforeData = data.before.mean_predicted.map((val, idx) => ({
        pred: val,
        actual: data.before.actual_default_rate[idx]
    }));
    
    const afterData = data.after.mean_predicted.map((val, idx) => ({
        pred: val,
        actual: data.after.actual_default_rate[idx]
    }));
    
    // Line generators
    const lineGen = d3.line()
        .x(d => xScale(d.pred))
        .y(d => yScale(d.actual));
        
    // Draw Before Calibration line (Red)
    chartGroup.append("path")
        .datum(beforeData)
        .attr("class", "calib-line-before")
        .attr("d", lineGen)
        .style("fill", "none")
        .style("stroke", "var(--accent-crimson)")
        .style("stroke-width", "2.5px");
        
    // Draw Before Calibration dots
    chartGroup.selectAll(".dot-before")
        .data(beforeData)
        .enter()
        .append("circle")
        .attr("class", "dot-before")
        .attr("cx", d => xScale(d.pred))
        .attr("cy", d => yScale(d.actual))
        .attr("r", 4)
        .style("fill", "var(--accent-crimson)")
        .on("mouseover", function(event, d) {
            showTooltip(event, `<strong>Before Calibration:</strong><br>Predicted: ${(d.pred*100).toFixed(1)}%<br>Actual Default: ${(d.actual*100).toFixed(1)}%`);
        })
        .on("mouseout", hideTooltip);
        
    // Draw After Calibration line (Teal)
    chartGroup.append("path")
        .datum(afterData)
        .attr("class", "calib-line-after")
        .attr("d", lineGen)
        .style("fill", "none")
        .style("stroke", "var(--accent-teal)")
        .style("stroke-width", "2.5px");
        
    // Draw After Calibration dots
    chartGroup.selectAll(".dot-after")
        .data(afterData)
        .enter()
        .append("circle")
        .attr("class", "dot-after")
        .attr("cx", d => xScale(d.pred))
        .attr("cy", d => yScale(d.actual))
        .attr("r", 4)
        .style("fill", "var(--accent-teal)")
        .on("mouseover", function(event, d) {
            showTooltip(event, `<strong>After Platt Scaling:</strong><br>Predicted: ${(d.pred*100).toFixed(1)}%<br>Actual Default: ${(d.actual*100).toFixed(1)}%`);
        })
        .on("mouseout", hideTooltip);
        
    // Draw Axis Title labels
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("x", chartWidth / 2)
        .attr("y", chartHeight + 35)
        .attr("text-anchor", "middle")
        .text("Predicted Default Risk (%)");
        
    chartGroup.append("text")
        .attr("class", "axis-label")
        .attr("transform", "rotate(-90)")
        .attr("x", -chartHeight / 2)
        .attr("y", -38)
        .attr("text-anchor", "middle")
        .text("Actual Default Rate (%)");
        
    // Draw Legend
    const legend = chartGroup.append("g")
        .attr("transform", `translate(20, 20)`);
        
    // Perfect Calibration line
    legend.append("line")
        .attr("x1", 0).attr("y1", 5).attr("x2", 15).attr("y2", 5)
        .style("stroke", "#64748b").style("stroke-dasharray", "3, 3").style("stroke-width", "1.5px");
    legend.append("text").attr("x", 22).attr("y", 9).text("Perfect Calibration").style("font-size", "0.7rem").style("fill", "var(--text-secondary)");
    
    // Before line
    legend.append("line")
        .attr("x1", 0).attr("y1", 20).attr("x2", 15).attr("y2", 20)
        .style("stroke", "var(--accent-crimson)").style("stroke-width", "2px");
    legend.append("text").attr("x", 22).attr("y", 24).text("Before Calibration").style("font-size", "0.7rem").style("fill", "var(--text-secondary)");
    
    // After line
    legend.append("line")
        .attr("x1", 0).attr("y1", 35).attr("x2", 15).attr("y2", 35)
        .style("stroke", "var(--accent-teal)").style("stroke-width", "2px");
    legend.append("text").attr("x", 22).attr("y", 39).text("After Calibration (Platt)").style("font-size", "0.7rem").style("fill", "var(--text-secondary)");
    
    svg.selectAll(".domain").style("stroke", "var(--chart-axis)");
    svg.selectAll(".tick line").style("stroke", "var(--chart-axis)");
}

// Utility to animate number counts smoothly
function animateValue(id, start, end, duration, suffix = "") {
    if (isNaN(start)) start = 0;
    if (start === end) {
        document.getElementById(id).textContent = `${(end * 100).toFixed(1)}${suffix}`;
        return;
    }
    
    const range = end - start;
    let current = start;
    const increment = end > start ? 0.01 : -0.01;
    const stepTime = Math.abs(Math.floor(duration / (range / increment)));
    
    const obj = document.getElementById(id);
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const val = start + range * progress;
        
        obj.textContent = `${(val * 100).toFixed(1)}${suffix}`;
        
        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            obj.textContent = `${(end * 100).toFixed(1)}${suffix}`;
        }
    }
    
    requestAnimationFrame(update);
}

// Tooltip display helpers
function showTooltip(event, htmlContent) {
    const tooltip = document.getElementById("chart-tooltip");
    tooltip.innerHTML = htmlContent;
    tooltip.style.opacity = 1;
    
    const x = event.pageX + 15;
    const y = event.pageY - 15;
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
}

function hideTooltip() {
    const tooltip = document.getElementById("chart-tooltip");
    tooltip.style.opacity = 0;
}

// Initialize Glossary drawer items
function initGlossary() {
    const listContainer = document.getElementById("glossary-list");
    listContainer.innerHTML = "";
    
    const sortedKeys = Object.keys(FEATURE_DISPLAY_NAMES).sort((a, b) => {
        return FEATURE_DISPLAY_NAMES[a].localeCompare(FEATURE_DISPLAY_NAMES[b]);
    });
    
    sortedKeys.forEach(key => {
        const item = document.createElement("div");
        item.className = "glossary-item";
        item.setAttribute("data-feature", key.toLowerCase());
        item.setAttribute("data-display", FEATURE_DISPLAY_NAMES[key].toLowerCase());
        
        item.innerHTML = `
            <div class="glossary-item-title">${FEATURE_DISPLAY_NAMES[key]}</div>
            <div class="glossary-item-desc">${FEATURE_DESCRIPTIONS[key]}</div>
            <div class="glossary-item-meta">Variable: ${key}</div>
        `;
        listContainer.appendChild(item);
    });
}

function toggleGlossary() {
    isGlossaryOpen = !isGlossaryOpen;
    const drawer = document.getElementById("glossary-drawer");
    if (isGlossaryOpen) {
        drawer.classList.add("open");
    } else {
        drawer.classList.remove("open");
    }
}

function filterGlossary(query) {
    const q = query.toLowerCase();
    const items = document.querySelectorAll(".glossary-item");
    items.forEach(item => {
        const title = item.getAttribute("data-display");
        const meta = item.getAttribute("data-feature");
        if (title.includes(q) || meta.includes(q)) {
            item.style.display = "block";
        } else {
            item.style.display = "none";
        }
    });
}

// ==========================================================================
// Authentication & Product Features implementation
// ==========================================================================

async function submitLogin() {
    const userEl = document.getElementById("login-username");
    const passEl = document.getElementById("login-password");
    const errorMsgEl = document.getElementById("login-error-msg");
    const errorTxtEl = document.getElementById("login-error-text");
    
    errorMsgEl.style.display = "none";
    
    try {
        const response = await fetch("/api/auth/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: userEl.value.trim(),
                password: passEl.value
            })
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Invalid credentials.");
        }
        
        const data = await response.json();
        localStorage.setItem("token", data.token);
        localStorage.setItem("user", JSON.stringify(data.user));
        sessionToken = data.token;
        
        userEl.value = "";
        passEl.value = "";
        
        await initDashboard();
    } catch (e) {
        errorTxtEl.textContent = e.message;
        errorMsgEl.style.display = "flex";
    }
}

async function logoutUser() {
    const token = localStorage.getItem("token");
    if (token) {
        try {
            await fetch("/api/auth/logout", {
                method: "POST",
                headers: { "Authorization": `Bearer ${token}` }
            });
        } catch (e) {
            console.error("Error during logout:", e);
        }
    }
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    sessionToken = null;
    currentPredictionId = null;
    
    document.getElementById("sidebar-user").style.display = "none";
    window.location.href = "/login";
}

let lookupTimeout = null;
async function handleGlobalLookup(query) {
    clearTimeout(lookupTimeout);
    const dropdown = document.getElementById("global-lookup-results");
    
    if (!query || query.trim().length < 2) {
        dropdown.style.display = "none";
        return;
    }
    
    lookupTimeout = setTimeout(async () => {
        try {
            const r = await authorizedFetch(`/applicants/search?query=${encodeURIComponent(query)}`);
            const matches = await r.json();
            
            if (matches.length === 0) {
                dropdown.innerHTML = '<div class="lookup-item" style="cursor:default;color:var(--text-muted);">No matching ID found</div>';
            } else {
                dropdown.innerHTML = matches.map(m => `
                    <div class="lookup-item" onclick="loadSearchedApplicant(${m.id})">
                        <i class="fa-solid fa-address-card" style="margin-right: 6px; color: var(--accent-blue);"></i>
                        <strong>${m.name}</strong> (ID: ${m.id})
                    </div>
                `).join('');
            }
            dropdown.style.display = "block";
        } catch (e) {
            console.error("Lookup error:", e);
        }
    }, 250);
}

async function loadSearchedApplicant(id) {
    document.getElementById("global-lookup-results").style.display = "none";
    document.getElementById("global-lookup-search").value = "";
    
    try {
        const r = await authorizedFetch(`/applicants/${id}`);
        const data = await r.json();
        
        const select = document.getElementById("form-preset-select");
        let optionExists = false;
        for (let i = 0; i < select.options.length; i++) {
            if (parseInt(select.options[i].value) === id) {
                optionExists = true;
                break;
            }
        }
        
        if (!optionExists) {
            const opt = document.createElement("option");
            opt.value = id;
            opt.textContent = `${data.name} (ID: ${id}) - ${data.target === 1 ? 'Default' : 'Paid'}`;
            select.insertBefore(opt, select.firstChild);
        }
        
        select.value = id;
        await loadFormPreset(id);
        switchTab("scoring");
    } catch (e) {
        console.error("Failed to load searched applicant:", e);
    }
}

async function submitUnderwritingDecision() {
    if (!currentPredictionId) {
        alert("Please run a risk assessment first before submitting a decision.");
        return;
    }
    
    const decisionVal = document.querySelector('input[name="underwriting-decision"]:checked').value;
    const notesVal = document.getElementById("underwriting-notes").value;
    
    try {
        const r = await authorizedFetch(`/api/applicant/${activeApplicantId}/decision`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                prediction_id: currentPredictionId,
                decision: decisionVal,
                notes: notesVal
            })
        });
        
        if (r.ok) {
            document.getElementById("underwriting-notes").value = "";
            await fetchDecisionHistory(activeApplicantId);
            await fetchOverrideStats();
            await loadDashboardHome();
        } else {
            const err = await r.json();
            alert("Error submitting decision: " + err.detail);
        }
    } catch (e) {
        console.error("Decision submission error:", e);
    }
}

async function fetchDecisionHistory(applicantId) {
    const timelineList = document.getElementById("timeline-list");
    timelineList.innerHTML = '<div style="text-align:center;padding:10px;"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</div>';
    
    try {
        const r = await authorizedFetch(`/api/applicant/${applicantId}/history`);
        const history = await r.json();
        
        if (history.length === 0) {
            timelineList.innerHTML = '<div class="timeline-empty">No previous evaluations logged.</div>';
            return;
        }
        
        timelineList.innerHTML = history.map(item => {
            const decision = item.decision;
            const badgeClass = decision === 'approved' ? 'approved' : (decision === 'declined' ? 'declined' : 'escalated');
            const timeStr = new Date(item.decision_time || item.pred_time).toLocaleString();
            
            let contentHtml = '';
            if (decision) {
                contentHtml = `
                    <div class="timeline-header">
                        <span class="timeline-badge ${badgeClass}">${decision}</span>
                        <span class="timeline-officer">by ${item.officer_name || 'System'}</span>
                    </div>
                    <div class="timeline-notes">${item.notes || 'No underwriting notes.'}</div>
                    ${item.narration ? `<div class="timeline-narration-sub" style="font-size: 11px; margin-top: 6px; padding-top: 6px; border-top: 1px dashed rgba(255,255,255,0.06); color: var(--text-secondary);"><i class="fa-solid fa-robot"></i> ${item.narration}</div>` : ''}
                `;
            } else {
                contentHtml = `
                    <div class="timeline-header">
                        <span class="timeline-officer" style="font-weight:600">Model Scored: ${(item.predicted_probability * 100).toFixed(1)}%</span>
                        <span class="timeline-officer">Risk: ${item.risk_label}</span>
                    </div>
                    <div class="timeline-notes">${item.narration ? `<i class="fa-solid fa-robot" style="font-size: 11px; margin-right: 4px; color: var(--accent-teal);"></i> ${item.narration}` : 'Awaiting underwriting officer decision.'}</div>
                `;
            }
            
            return `
                <div class="timeline-item ${badgeClass}">
                    ${contentHtml}
                    <div class="timeline-time">${timeStr}</div>
                </div>
            `;
        }).join('');
    } catch (e) {
        console.error("Failed to load history:", e);
        timelineList.innerHTML = '<div class="timeline-empty">Error loading evaluations timeline.</div>';
    }
}

async function fetchOverrideStats() {
    try {
        const r = await authorizedFetch("/api/decisions/stats");
        const stats = await r.json();
        
        document.getElementById("override-stat-total").textContent = stats.total_decisions;
        document.getElementById("override-stat-rate").textContent = `${(stats.override_rate * 100).toFixed(1)}%`;
        document.getElementById("override-stat-count").textContent = stats.overrides_count;
    } catch (e) {
        console.error("Failed to fetch override statistics:", e);
    }
}

// Click outside lookup results closes dropdown
document.addEventListener("click", (e) => {
    const dropdown = document.getElementById("global-lookup-results");
    const searchInput = document.getElementById("global-lookup-search");
    if (dropdown && e.target !== searchInput && !dropdown.contains(e.target)) {
        dropdown.style.display = "none";
    }
});

// AI Chatbot State
let chatHistory = [];

function resetChatForApplicant(applicantId) {
    chatHistory = [];
    const chatMessages = document.getElementById("ai-chat-messages");
    if (chatMessages) {
        chatMessages.innerHTML = `
            <div class="chat-message system">
                Welcome! I'm your AI Underwriting Assistant. Ask me questions about risk factors and loan metrics for Applicant #${applicantId}.
            </div>
        `;
    }
    const pulseBadge = document.getElementById("chat-pulse");
    if (pulseBadge) {
        pulseBadge.style.display = "block";
    }
}

// Initialize chatbot event listeners
document.addEventListener("DOMContentLoaded", () => {
    const chatBubble = document.getElementById("ai-chat-bubble");
    const chatWindow = document.getElementById("ai-chat-window");
    const chatClose = document.getElementById("ai-chat-close");
    const chatForm = document.getElementById("ai-chat-form");
    const chatInput = document.getElementById("ai-chat-input");
    const chatMessages = document.getElementById("ai-chat-messages");
    const pulseBadge = document.getElementById("chat-pulse");

    if (!chatBubble || !chatWindow) return;

    // Toggle Chat window
    chatBubble.addEventListener("click", () => {
        chatWindow.classList.toggle("hidden");
        if (pulseBadge) pulseBadge.style.display = "none"; // Hide alert pulse once opened
        // Auto scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });

    chatClose.addEventListener("click", () => {
        chatWindow.classList.add("hidden");
    });

    // Form submit
    chatForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const text = chatInput.value.trim();
        if (!text) return;

        chatInput.value = "";
        appendChatMessage("user", text);

        // Append typing loader
        const loadingDiv = document.createElement("div");
        loadingDiv.className = "chat-message ai loading";
        loadingDiv.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...`;
        chatMessages.appendChild(loadingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        const newMessages = [...chatHistory, { role: "user", content: text }];

        try {
            const response = await authorizedFetch("/chat", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    applicant_id: activeApplicantId ? parseInt(activeApplicantId) : null,
                    messages: newMessages
                })
            });

            // Remove loading
            if (loadingDiv.parentNode) {
                chatMessages.removeChild(loadingDiv);
            }

            if (response.ok) {
                const data = await response.json();
                appendChatMessage("ai", data.reply);
                // Keep history updated with structured messages
                chatHistory = [...newMessages, { role: "assistant", content: data.reply }];
                // Cap history size to keep payload lightweight (keep last 20 messages)
                if (chatHistory.length > 20) {
                    chatHistory = chatHistory.slice(-20);
                }
            } else {
                appendChatMessage("ai", "I'm sorry, I couldn't process that message. Please try again.");
            }
        } catch (error) {
            console.error("Chat error:", error);
            if (loadingDiv.parentNode) {
                chatMessages.removeChild(loadingDiv);
            }
            appendChatMessage("ai", "Network error. Make sure your local server is running.");
        }
    });

    // Bind suggestion buttons
    const suggestBtns = document.querySelectorAll(".chat-suggest-btn");
    suggestBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            chatInput.value = btn.textContent;
            chatForm.dispatchEvent(new Event("submit"));
        });
    });

    function appendChatMessage(role, text) {
        const div = document.createElement("div");
        div.className = `chat-message ${role}`;
        div.textContent = text;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});

// Theme switcher functions
function initTheme() {
    const savedTheme = localStorage.getItem("theme") || "dark";
    const body = document.body;
    const darkIcon = document.querySelector(".theme-icon-dark");
    const lightIcon = document.querySelector(".theme-icon-light");
    const themeText = document.getElementById("theme-text");
    
    if (savedTheme === "light") {
        body.classList.remove("dark-theme");
        body.classList.add("light-theme");
        if (darkIcon) darkIcon.style.display = "none";
        if (lightIcon) lightIcon.style.display = "inline-block";
        if (themeText) themeText.textContent = "Light";
    } else {
        body.classList.remove("light-theme");
        body.classList.add("dark-theme");
        if (darkIcon) darkIcon.style.display = "inline-block";
        if (lightIcon) lightIcon.style.display = "none";
        if (themeText) themeText.textContent = "Dark";
    }
}

function toggleTheme() {
    const body = document.body;
    const darkIcon = document.querySelector(".theme-icon-dark");
    const lightIcon = document.querySelector(".theme-icon-light");
    const themeText = document.getElementById("theme-text");
    
    if (body.classList.contains("dark-theme")) {
        body.classList.remove("dark-theme");
        body.classList.add("light-theme");
        localStorage.setItem("theme", "light");
        if (darkIcon) darkIcon.style.display = "none";
        if (lightIcon) lightIcon.style.display = "inline-block";
        if (themeText) themeText.textContent = "Light";
    } else {
        body.classList.remove("light-theme");
        body.classList.add("dark-theme");
        localStorage.setItem("theme", "dark");
        if (darkIcon) darkIcon.style.display = "inline-block";
        if (lightIcon) lightIcon.style.display = "none";
        if (themeText) themeText.textContent = "Dark";
    }
}
