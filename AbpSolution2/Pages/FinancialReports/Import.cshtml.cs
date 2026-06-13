using AbpSolution2.Permissions;
using Microsoft.AspNetCore.Authorization;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using AbpSolution2.Data;
using AbpSolution2.FinancialReports;
using CsvHelper;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Volo.Abp.AspNetCore.Mvc.UI.RazorPages;
using Volo.Abp.Uow;

namespace AbpSolution2.Pages.FinancialReports
{
    [Authorize(AbpSolution2Permissions.FinancialReports.Import)]
    public class ImportModel : AbpPageModel
    {
        private readonly AbpSolution2DbContext _dbContext;

        [BindProperty]
        public IFormFile? CsvFile { get; set; }

        [TempData]
        public string? StatusMessage { get; set; }

        public ImportModel(AbpSolution2DbContext dbContext)
        {
            _dbContext = dbContext;
        }

        public void OnGet()
        {
        }

        [UnitOfWork]
        public async Task<IActionResult> OnPostAsync()
        {
            if (CsvFile == null || CsvFile.Length == 0)
            {
                StatusMessage = "請選擇 CSV 檔案。";
                return Page();
            }

            int importedCount = 0;
            int skippedCount = 0;

            using var stream = CsvFile.OpenReadStream();
            using var reader = new StreamReader(stream, Encoding.UTF8);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);

            var records = csv.GetRecords<dynamic>();

            foreach (var record in records)
            {
                var dict = (IDictionary<string, object>)record;

                int year = ToInt(dict, "年度");
                int quarter = ToInt(dict, "季別");
                string companyCode = ToStringValue(dict, "公司代號");

                if (string.IsNullOrWhiteSpace(companyCode))
                {
                    skippedCount++;
                    continue;
                }

                bool exists = await _dbContext.FinancialReports.AnyAsync(x =>
                    x.Year == year &&
                    x.Quarter == quarter &&
                    x.CompanyCode == companyCode);

                if (exists)
                {
                    skippedCount++;
                    continue;
                }

                var report = new FinancialReport(
                    Guid.NewGuid(),
                    ToStringValue(dict, "出表日期"),
                    year,
                    quarter,
                    companyCode,
                    ToStringValue(dict, "公司名稱"),
                    ToDecimal(dict, "資產總計"),
                    ToDecimal(dict, "負債總計"),
                    ToDecimal(dict, "權益總計"),
                    ToDecimal(dict, "每股參考淨值"),
                    JsonSerializer.Serialize(dict)
                );

                _dbContext.FinancialReports.Add(report);
                importedCount++;
            }

            await _dbContext.SaveChangesAsync();

            StatusMessage = $"匯入完成：新增 {importedCount} 筆，略過 {skippedCount} 筆。";

            return RedirectToPage("/FinancialReports/Index");
        }

        private static string ToStringValue(IDictionary<string, object> dict, string key)
        {
            if (!dict.ContainsKey(key) || dict[key] == null)
            {
                return string.Empty;
            }

            return dict[key]?.ToString()?.Trim() ?? string.Empty;
        }

        private static int ToInt(IDictionary<string, object> dict, string key)
        {
            string value = ToStringValue(dict, key);

            if (int.TryParse(value, out int result))
            {
                return result;
            }

            return 0;
        }

        private static decimal? ToDecimal(IDictionary<string, object> dict, string key)
        {
            string value = ToStringValue(dict, key);

            if (string.IsNullOrWhiteSpace(value))
            {
                return null;
            }

            value = value.Replace(",", "");

            if (decimal.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out decimal result))
            {
                return result;
            }

            return null;
        }
    }
}