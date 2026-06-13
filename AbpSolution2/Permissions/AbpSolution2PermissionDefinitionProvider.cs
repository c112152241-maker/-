using AbpSolution2.Localization;
using Volo.Abp.Authorization.Permissions;
using Volo.Abp.Localization;
using Volo.Abp.MultiTenancy;

namespace AbpSolution2.Permissions;

public class AbpSolution2PermissionDefinitionProvider : PermissionDefinitionProvider
{
    public override void Define(IPermissionDefinitionContext context)
    {
        var myGroup = context.AddGroup(AbpSolution2Permissions.GroupName);

        var financialReports = myGroup.AddPermission(
            AbpSolution2Permissions.FinancialReports.Default,
            L("Permission:FinancialReports"));

        financialReports.AddChild(
            AbpSolution2Permissions.FinancialReports.Import,
            L("Permission:FinancialReports.Import"));

        financialReports.AddChild(
            AbpSolution2Permissions.FinancialReports.Export,
            L("Permission:FinancialReports.Export"));

        financialReports.AddChild(
            AbpSolution2Permissions.FinancialReports.Delete,
            L("Permission:FinancialReports.Delete"));

        financialReports.AddChild(
            AbpSolution2Permissions.FinancialReports.Analysis,
            L("Permission:FinancialReports.Analysis"));



        //Define your own permissions here. Example:
        //myGroup.AddPermission(AbpSolution2Permissions.MyPermission1, L("Permission:MyPermission1"));
    }

    private static LocalizableString L(string name)
    {
        return LocalizableString.Create<AbpSolution2Resource>(name);
    }
}
