using AbpSolution2.Permissions;
using Microsoft.AspNetCore.Authorization;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AbpSolution2.Data;
using AbpSolution2.FinancialReports;
using Microsoft.EntityFrameworkCore;
using Volo.Abp.AspNetCore.Mvc.UI.RazorPages;
using Volo.Abp.Uow;

namespace AbpSolution2.Pages.FinancialReports
{
    [Authorize(AbpSolution2Permissions.FinancialReports.Analysis)]
    public class AnalysisModel : AbpPageModel
    {
        private readonly AbpSolution2DbContext _dbContext;

        public int TotalCompanyCount { get; set; }

        public FinancialReport? TopAssetCompany { get; set; }

        public FinancialReport? TopDebtRatioCompany { get; set; }

        public FinancialReport? TopNetWorthCompany { get; set; }

        public List<FinancialReport> AssetRanking { get; set; } = new();

        public List<FinancialReport> DebtRatioRanking { get; set; } = new();

        public List<FinancialReport> NetWorthRanking { get; set; } = new();

        public List<string> AssetChartLabels { get; set; } = new();

        public List<decimal> AssetChartValues { get; set; } = new();

        public List<string> DebtRatioChartLabels { get; set; } = new();

        public List<decimal> DebtRatioChartValues { get; set; } = new();

        public List<string> NetWorthChartLabels { get; set; } = new();

        public List<decimal> NetWorthChartValues { get; set; } = new();

        public AnalysisModel(AbpSolution2DbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [UnitOfWork]
        public async Task OnGetAsync()
        {
            var reports = await _dbContext.FinancialReports
                .OrderBy(x => x.CompanyCode)
                .ToListAsync();

            TotalCompanyCount = reports.Count;

            TopAssetCompany = reports
                .OrderByDescending(x => x.TotalAssets ?? 0)
                .FirstOrDefault();

            TopDebtRatioCompany = reports
                .Where(x => x.DebtRatio != null)
                .OrderByDescending(x => x.DebtRatio)
                .FirstOrDefault();

            TopNetWorthCompany = reports
                .OrderByDescending(x => x.NetWorthPerShare ?? 0)
                .FirstOrDefault();

            AssetRanking = reports
                .OrderByDescending(x => x.TotalAssets ?? 0)
                .Take(5)
                .ToList();

            DebtRatioRanking = reports
                .Where(x => x.DebtRatio != null)
                .OrderByDescending(x => x.DebtRatio)
                .Take(5)
                .ToList();

            NetWorthRanking = reports
                .OrderByDescending(x => x.NetWorthPerShare ?? 0)
                .Take(5)
                .ToList();

            AssetChartLabels = AssetRanking.Select(x => x.CompanyName).ToList();
            AssetChartValues = AssetRanking.Select(x => x.TotalAssets ?? 0).ToList();

            DebtRatioChartLabels = DebtRatioRanking.Select(x => x.CompanyName).ToList();
            DebtRatioChartValues = DebtRatioRanking.Select(x => x.DebtRatio ?? 0).ToList();

            NetWorthChartLabels = NetWorthRanking.Select(x => x.CompanyName).ToList();
            NetWorthChartValues = NetWorthRanking.Select(x => x.NetWorthPerShare ?? 0).ToList();
        }
    }
}