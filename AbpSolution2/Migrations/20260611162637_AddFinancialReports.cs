using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace AbpSolution2.Migrations
{
    /// <inheritdoc />
    public partial class AddFinancialReports : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "AppFinancialReports",
                columns: table => new
                {
                    Id = table.Column<Guid>(type: "TEXT", nullable: false),
                    ReportDate = table.Column<string>(type: "TEXT", maxLength: 50, nullable: false),
                    Year = table.Column<int>(type: "INTEGER", nullable: false),
                    Quarter = table.Column<int>(type: "INTEGER", nullable: false),
                    CompanyCode = table.Column<string>(type: "TEXT", maxLength: 20, nullable: false),
                    CompanyName = table.Column<string>(type: "TEXT", maxLength: 100, nullable: false),
                    TotalAssets = table.Column<decimal>(type: "TEXT", precision: 28, scale: 2, nullable: true),
                    TotalLiabilities = table.Column<decimal>(type: "TEXT", precision: 28, scale: 2, nullable: true),
                    TotalEquity = table.Column<decimal>(type: "TEXT", precision: 28, scale: 2, nullable: true),
                    NetWorthPerShare = table.Column<decimal>(type: "TEXT", precision: 28, scale: 2, nullable: true),
                    RawJson = table.Column<string>(type: "TEXT", nullable: true),
                    ExtraProperties = table.Column<string>(type: "TEXT", nullable: false),
                    ConcurrencyStamp = table.Column<string>(type: "TEXT", maxLength: 40, nullable: false),
                    CreationTime = table.Column<DateTime>(type: "TEXT", nullable: false),
                    CreatorId = table.Column<Guid>(type: "TEXT", nullable: true),
                    LastModificationTime = table.Column<DateTime>(type: "TEXT", nullable: true),
                    LastModifierId = table.Column<Guid>(type: "TEXT", nullable: true),
                    IsDeleted = table.Column<bool>(type: "INTEGER", nullable: false, defaultValue: false),
                    DeleterId = table.Column<Guid>(type: "TEXT", nullable: true),
                    DeletionTime = table.Column<DateTime>(type: "TEXT", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_AppFinancialReports", x => x.Id);
                });

            migrationBuilder.CreateIndex(
                name: "IX_AppFinancialReports_CompanyName",
                table: "AppFinancialReports",
                column: "CompanyName");

            migrationBuilder.CreateIndex(
                name: "IX_AppFinancialReports_NetWorthPerShare",
                table: "AppFinancialReports",
                column: "NetWorthPerShare");

            migrationBuilder.CreateIndex(
                name: "IX_AppFinancialReports_TotalAssets",
                table: "AppFinancialReports",
                column: "TotalAssets");

            migrationBuilder.CreateIndex(
                name: "IX_AppFinancialReports_Year_Quarter_CompanyCode",
                table: "AppFinancialReports",
                columns: new[] { "Year", "Quarter", "CompanyCode" },
                unique: true);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "AppFinancialReports");
        }
    }
}
