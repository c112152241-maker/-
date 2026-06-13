using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Design;

namespace AbpSolution2.Data;

public class AbpSolution2DbContextFactory : IDesignTimeDbContextFactory<AbpSolution2DbContext>
{
    public AbpSolution2DbContext CreateDbContext(string[] args)
    {
        AbpSolution2GlobalFeatureConfigurator.Configure();
        AbpSolution2ModuleExtensionConfigurator.Configure();

        AbpSolution2EfCoreEntityExtensionMappings.Configure();
        var configuration = BuildConfiguration();

        var builder = new DbContextOptionsBuilder<AbpSolution2DbContext>()
            .UseSqlite(configuration.GetConnectionString("Default"));

        return new AbpSolution2DbContext(builder.Options);
    }

    private static IConfigurationRoot BuildConfiguration()
    {
        var builder = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false)
            .AddEnvironmentVariables();

        return builder.Build();
    }
}