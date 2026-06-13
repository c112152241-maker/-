using AbpSolution2.Permissions;
using Microsoft.AspNetCore.Authorization;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AbpSolution2.Data;
using AbpSolution2.FinancialReports;
using ClosedXML.Excel;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Volo.Abp.AspNetCore.Mvc.UI.RazorPages;
using Volo.Abp.Uow;

namespace AbpSolution2.Pages.FinancialReports
{
    [Authorize(AbpSolution2Permissions.FinancialReports.Default)]
    public class IndexModel : AbpPageModel
    {
        private readonly AbpSolution2DbContext _dbContext;

        private readonly IAuthorizationService _authorizationService;
        public IndexModel(
            AbpSolution2DbContext dbContext,
            IAuthorizationService authorizationService)
                {
                    _dbContext = dbContext;
                    _authorizationService = authorizationService;
                }

        public List<FinancialReport> Reports { get; set; } = new();

        [TempData]
        public string? StatusMessage { get; set; }

        public IndexModel(AbpSolution2DbContext dbContext)
        {
            _dbContext = dbContext;
        }

        [UnitOfWork]
        public async Task OnGetAsync()
        {
            Reports = await _dbContext.FinancialReports
                .OrderBy(x => x.CompanyCode)
                .ToListAsync();
        }

        [UnitOfWork]
        public async Task<IActionResult> OnPostClearAsync()
        {
            var clearAuthorization = await _authorizationService.AuthorizeAsync(
                User,
                AbpSolution2Permissions.FinancialReports.Delete);

            if (!clearAuthorization.Succeeded)
            {
                return Forbid();
            }

            var reports = await _dbContext.FinancialReports.ToListAsync();

            _dbContext.FinancialReports.RemoveRange(reports);

            await _dbContext.SaveChangesAsync();

            StatusMessage = $"已清空 {reports.Count} 筆財報資料。";

            return RedirectToPage();
        }

        [UnitOfWork]
        public async Task<IActionResult> OnGetExportAsync()
        {
            var exportAuthorization = await _authorizationService.AuthorizeAsync(
                User,
                AbpSolution2Permissions.FinancialReports.Export);

            if (!exportAuthorization.Succeeded)
            {
                return Forbid();
            }

            var reports = await _dbContext.FinancialReports
                .OrderBy(x => x.CompanyCode)
                .ToListAsync();

            using var workbook = new XLWorkbook();
            var worksheet = workbook.Worksheets.Add("FinancialReports");

            worksheet.Cell(1, 1).Value = "年度";
            worksheet.Cell(1, 2).Value = "季別";
            worksheet.Cell(1, 3).Value = "公司代號";
            worksheet.Cell(1, 4).Value = "公司名稱";
            worksheet.Cell(1, 5).Value = "資產總計";
            worksheet.Cell(1, 6).Value = "負債總計";
            worksheet.Cell(1, 7).Value = "權益總計";
            worksheet.Cell(1, 8).Value = "負債比";
            worksheet.Cell(1, 9).Value = "每股參考淨值";

            var row = 2;

            foreach (var item in reports)
            {
                worksheet.Cell(row, 1).Value = item.Year;
                worksheet.Cell(row, 2).Value = item.Quarter;
                worksheet.Cell(row, 3).Value = item.CompanyCode;
                worksheet.Cell(row, 4).Value = item.CompanyName;
                worksheet.Cell(row, 5).Value = item.TotalAssets;
                worksheet.Cell(row, 6).Value = item.TotalLiabilities;
                worksheet.Cell(row, 7).Value = item.TotalEquity;
                worksheet.Cell(row, 8).Value = item.DebtRatio;
                worksheet.Cell(row, 9).Value = item.NetWorthPerShare;

                row++;
            }

            worksheet.Columns().AdjustToContents();

            using var stream = new MemoryStream();
            workbook.SaveAs(stream);

            var fileBytes = stream.ToArray();

            return File(
                fileBytes,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "FinancialReports.xlsx"
            );
        }
    }
}