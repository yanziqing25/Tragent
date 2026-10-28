import type { components } from './schema'

type Schemas = components['schemas']

export type AccountView = Schemas['AccountView']
export type BrokerCredentials = Schemas['BrokerCredentials']
export type BudgetUsage = Schemas['BudgetUsage']
export type CheckResult = Schemas['CheckResult']
export type CriticVerdict = Schemas['CriticVerdict']
export type DeploymentRun = Schemas['Run']
export type DeploymentView = Schemas['DeploymentView']
export type HistoryTrade = Schemas['HistoryTrade']
export type MarketContextEvidence = Schemas['MarketContextEvidence']
export type Order = Schemas['Order']
export type OrderRequest = Schemas['OrderRequest']
export type PerformanceMetrics = Schemas['PerformanceMetrics']
export type ReportCandidate = Schemas['ReportCandidate']
export type ReportCandidateSegment = Schemas['ReportCandidateSegment']
export type ReportReferenceSegment = Schemas['ReportReferenceSegment']
export type ReportTradeRow = Schemas['ReportTradeRow']
export type ResearchGoalView = Schemas['ResearchGoalView']
export type ResearchReportView = Schemas['ResearchReportView']
export type ResearchTaskSummary = Schemas['ResearchTaskSummary']
export type ResearchTaskView = Schemas['ResearchTaskView']
export type ResumeResearchRequest = Schemas['ResumeResearchRequest']
export type Statement = Schemas['Statement']
export type StrategyName = Schemas['StrategyName']
export type TaskStatus = Schemas['TaskStatus']

export type SegmentName = ReportCandidateSegment['segment']

// The SSE endpoint is documented as a generic event stream, so its payloads are typed here.
export type AgentStatusSnapshot = Pick<
  ResearchTaskView,
  | 'active_agent'
  | 'budget_used'
  | 'budget_remaining'
  | 'critic_round'
  | 'critic_verdict'
  | 'critic_budget_used'
  | 'critic_budget_remaining'
>

export interface AgentEvent {
  source: 'goal_parser' | 'workflow' | 'research_agent' | 'critic_agent'
  kind: 'status' | 'decision' | 'tool' | 'observation'
  message: string
  tool_name: string | null
  agent_status: AgentStatusSnapshot | null
}

export type TaskStreamEvent =
  | { type: 'agent'; data: AgentEvent }
  | { type: 'result'; data: ResearchTaskView }
  | { type: 'error'; data: { detail: string } }

// The trading WebSocket is not in the OpenAPI schema, so its messages are typed here.
export type AccountSnapshot = Pick<AccountView, 'balances' | 'positions' | 'synced_at'>

export type TradingMessage =
  { type: 'order'; data: Order } | { type: 'account'; data: AccountSnapshot }
