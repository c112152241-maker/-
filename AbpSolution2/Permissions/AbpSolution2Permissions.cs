namespace AbpSolution2.Permissions;

public static class AbpSolution2Permissions
{
    public const string GroupName = "AbpSolution2";

    public static class FinancialReports
    {
        public const string Default = GroupName + ".FinancialReports";
        public const string Import = Default + ".Import";
        public const string Export = Default + ".Export";
        public const string Delete = Default + ".Delete";
        public const string Analysis = Default + ".Analysis";
    }

}
