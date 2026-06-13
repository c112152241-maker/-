using System;
using Volo.Abp.Domain.Entities.Auditing;

namespace AbpSolution2.FinancialReports
{
    public class FinancialReport : FullAuditedAggregateRoot<Guid>
    {
        public string ReportDate { get; private set; } = string.Empty;

        public int Year { get; private set; }

        public int Quarter { get; private set; }

        public string CompanyCode { get; private set; } = string.Empty;

        public string CompanyName { get; private set; } = string.Empty;

        public decimal? TotalAssets { get; private set; }

        public decimal? TotalLiabilities { get; private set; }

        public decimal? TotalEquity { get; private set; }

        public decimal? NetWorthPerShare { get; private set; }

        public string? RawJson { get; private set; }

        public decimal? DebtRatio
        {
            get
            {
                if (TotalAssets == null || TotalAssets == 0 || TotalLiabilities == null)
                {
                    return null;
                }

                return TotalLiabilities / TotalAssets * 100;
            }
        }

        protected FinancialReport()
        {
        }

        public FinancialReport(
            Guid id,
            string reportDate,
            int year,
            int quarter,
            string companyCode,
            string companyName,
            decimal? totalAssets,
            decimal? totalLiabilities,
            decimal? totalEquity,
            decimal? netWorthPerShare,
            string? rawJson)
            : base(id)
        {
            ReportDate = reportDate;
            Year = year;
            Quarter = quarter;
            CompanyCode = companyCode;
            CompanyName = companyName;
            TotalAssets = totalAssets;
            TotalLiabilities = totalLiabilities;
            TotalEquity = totalEquity;
            NetWorthPerShare = netWorthPerShare;
            RawJson = rawJson;
        }
    }
}