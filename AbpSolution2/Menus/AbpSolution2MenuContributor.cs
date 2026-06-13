using AbpSolution2.Permissions;
using AbpSolution2.Localization;
using Volo.Abp.Authorization.Permissions;
using Volo.Abp.Identity.Web.Navigation;
using Volo.Abp.SettingManagement.Web.Navigation;
using Volo.Abp.UI.Navigation;
using Volo.Abp.Identity.Web.Navigation;

namespace AbpSolution2.Menus;

public class AbpSolution2MenuContributor : IMenuContributor
{
    public async Task ConfigureMenuAsync(MenuConfigurationContext context)
    {
        if (context.Menu.Name == StandardMenus.Main)
        {
            await ConfigureMainMenuAsync(context);
        }
    }

    private static Task ConfigureMainMenuAsync(MenuConfigurationContext context)
    {
        var l = context.GetLocalizer<AbpSolution2Resource>();
        context.Menu.Items.Insert(
            0,
            new ApplicationMenuItem(
                AbpSolution2Menus.Home,
                l["Menu:Home"],
                "~/",
                icon: "fas fa-home",
                order: 0
            )
        );

        var financialReportsMenu = new ApplicationMenuItem(
            "FinancialReports",
            "財報系統",
            icon: "fas fa-chart-line"
        );

        financialReportsMenu.AddItem(
            new ApplicationMenuItem(
                "FinancialReports.List",
                "財報列表",
                url: "/FinancialReports",
                icon: "fas fa-table"
            )
        );

        financialReportsMenu.AddItem(
            new ApplicationMenuItem(
                "FinancialReports.Import",
                "匯入 CSV",
                url: "/FinancialReports/Import",
                icon: "fas fa-file-upload"
            )
        );

        financialReportsMenu.AddItem(
            new ApplicationMenuItem(
                "FinancialReports.Analysis",
                "財報分析",
                url: "/FinancialReports/Analysis",
                icon: "fas fa-chart-bar"
            )
        );

        context.Menu.AddItem(financialReportsMenu);


        //Administration
        var administration = context.Menu.GetAdministration();
        administration.Order = 5;
        //Administration->Identity
        administration.SetSubItemOrder(IdentityMenuNames.GroupName, 2);

        //Administration->Settings
        administration.SetSubItemOrder(SettingManagementMenuNames.GroupName, 8);
        
        return Task.CompletedTask;
    }
}
