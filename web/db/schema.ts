import {
  pgTable,
  text,
  uuid,
  timestamp,
  varchar,
  jsonb,
  pgEnum,
  index,
  uniqueIndex,
} from "drizzle-orm/pg-core";

// ===== Enums =====

export const riskLevelEnum = pgEnum("risk_level", [
  "prohibited",
  "high",
  "limited",
  "minimal",
  "gpai",
  "unclassified",
]);

export const userRoleEnum = pgEnum("user_role", ["admin", "editor", "viewer"]);

export const companySizeEnum = pgEnum("company_size", [
  "1-49",
  "50-249",
  "250-999",
  "1000+",
]);

// ===== Tables =====

export const organizations = pgTable("organizations", {
  id: uuid("id").defaultRandom().primaryKey(),
  name: varchar("name", { length: 200 }).notNull(),
  industry: varchar("industry", { length: 100 }),
  size: companySizeEnum("size"),
  country: varchar("country", { length: 2 }).default("DE"),
  createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
});

export const users = pgTable(
  "users",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    orgId: uuid("org_id").references(() => organizations.id, { onDelete: "cascade" }),
    email: varchar("email", { length: 320 }).notNull(),
    name: varchar("name", { length: 200 }),
    role: userRoleEnum("role").default("editor").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => ({
    emailIdx: uniqueIndex("users_email_idx").on(table.email),
    orgIdx: index("users_org_idx").on(table.orgId),
  }),
);

export const aiSystems = pgTable(
  "ai_systems",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    orgId: uuid("org_id")
      .notNull()
      .references(() => organizations.id, { onDelete: "cascade" }),
    name: varchar("name", { length: 200 }).notNull(),
    provider: varchar("provider", { length: 200 }),
    version: varchar("version", { length: 50 }),
    purpose: text("purpose"),
    department: varchar("department", { length: 100 }),
    deploymentType: varchar("deployment_type", { length: 50 }), // saas | api | onprem
    dataCategories: jsonb("data_categories").$type<string[]>().default([]),
    riskClassification: riskLevelEnum("risk_classification").default("unclassified"),
    classifiedAt: timestamp("classified_at", { withTimezone: true }),
    classifiedBy: uuid("classified_by").references(() => users.id),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
    updatedAt: timestamp("updated_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => ({
    orgIdx: index("ai_systems_org_idx").on(table.orgId),
  }),
);

export const classificationSessions = pgTable("classification_sessions", {
  id: uuid("id").defaultRandom().primaryKey(),
  aiSystemId: uuid("ai_system_id")
    .notNull()
    .references(() => aiSystems.id, { onDelete: "cascade" }),
  answers: jsonb("answers").$type<Record<string, string>>().notNull(),
  result: riskLevelEnum("result").notNull(),
  rationale: text("rationale"),
  legalReferences: jsonb("legal_references").$type<string[]>().default([]),
  performedBy: uuid("performed_by").references(() => users.id),
  performedAt: timestamp("performed_at", { withTimezone: true }).defaultNow().notNull(),
});

export const documents = pgTable(
  "documents",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    orgId: uuid("org_id")
      .notNull()
      .references(() => organizations.id, { onDelete: "cascade" }),
    aiSystemId: uuid("ai_system_id").references(() => aiSystems.id, {
      onDelete: "set null",
    }),
    templateKey: varchar("template_key", { length: 100 }).notNull(),
    version: varchar("version", { length: 20 }).notNull(),
    fileUrl: text("file_url"),
    generatedBy: uuid("generated_by").references(() => users.id),
    generatedAt: timestamp("generated_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => ({
    orgIdx: index("documents_org_idx").on(table.orgId),
  }),
);

export const trainingRecords = pgTable("training_records", {
  id: uuid("id").defaultRandom().primaryKey(),
  orgId: uuid("org_id")
    .notNull()
    .references(() => organizations.id, { onDelete: "cascade" }),
  employeeName: varchar("employee_name", { length: 200 }).notNull(),
  employeeEmail: varchar("employee_email", { length: 320 }).notNull(),
  courseKey: varchar("course_key", { length: 100 }).notNull(),
  completedAt: timestamp("completed_at", { withTimezone: true }),
  score: varchar("score", { length: 10 }),
  certificateUrl: text("certificate_url"),
});

export const auditEvents = pgTable(
  "audit_events",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    orgId: uuid("org_id")
      .notNull()
      .references(() => organizations.id, { onDelete: "cascade" }),
    userId: uuid("user_id").references(() => users.id),
    entityType: varchar("entity_type", { length: 50 }).notNull(),
    entityId: uuid("entity_id"),
    action: varchar("action", { length: 50 }).notNull(),
    before: jsonb("before"),
    after: jsonb("after"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => ({
    orgIdx: index("audit_events_org_idx").on(table.orgId),
  }),
);

// ===== Marketing / Public =====

export const waitlist = pgTable(
  "waitlist",
  {
    id: uuid("id").defaultRandom().primaryKey(),
    email: varchar("email", { length: 320 }).notNull(),
    company: varchar("company", { length: 200 }).notNull(),
    companySize: companySizeEnum("company_size").notNull(),
    source: varchar("source", { length: 50 }).default("landing-page"),
    createdAt: timestamp("created_at", { withTimezone: true }).defaultNow().notNull(),
  },
  (table) => ({
    emailIdx: uniqueIndex("waitlist_email_idx").on(table.email),
  }),
);
