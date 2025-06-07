CREATE TABLE "reports" (
	"id" serial PRIMARY KEY NOT NULL,
	"title" text NOT NULL,
	"content" text NOT NULL,
	"total_articles" integer NOT NULL,
	"total_sources" integer NOT NULL,
	"used_articles" integer NOT NULL,
	"used_sources" integer NOT NULL,
	"created_at" timestamp DEFAULT CURRENT_TIMESTAMP
);
