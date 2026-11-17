using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;
using System.Collections.Generic;

[assembly: System.Reflection.AssemblyTitle("PurposeBuilt Scan Tools")]
[assembly: System.Reflection.AssemblyVersion("1.0.0.0")]
[assembly: System.Reflection.AssemblyFileVersion("1.0.0.0")]

internal static class Program {
    [STAThread]
    private static int Main(string[] args) {
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        using (var runtime = new ScanRuntime()) {
            if (args.Length == 1 && args[0] == "--self-test") {
                try { Console.WriteLine(runtime.Run("self-test", null, null)); return 0; }
                catch (Exception e) { Console.Error.WriteLine(e.Message); return 1; }
            }
            Application.Run(new ScanForm(runtime));
        }
        return 0;
    }
}

internal sealed class ScanRuntime : IDisposable {
    internal readonly string Resources = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "resources");
    internal readonly string Support = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "PurposeBuilt Scan Tools");
    internal readonly string Scans = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "PurposeBuilt Scans");
    private Process worker;
    private readonly IntPtr job;
    private readonly JavaScriptSerializer json = new JavaScriptSerializer();
    internal ScanRuntime() {
        job = CreateJobObject(IntPtr.Zero, null);
        if (job == IntPtr.Zero) throw new System.ComponentModel.Win32Exception();
        var limits = new JobLimits();
        limits.Basic.LimitFlags = 0x2000; // Kill this launcher's worker family when it exits.
        if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(limits))) throw new System.ComponentModel.Win32Exception();
    }
    private static string Quote(string value) {
        var result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char character in value) {
            if (character == '\\') { slashes++; continue; }
            if (character == '"') { result.Append('\\', slashes * 2 + 1); result.Append(character); slashes = 0; continue; }
            result.Append('\\', slashes); slashes = 0; result.Append(character);
        }
        result.Append('\\', slashes * 2); result.Append('"'); return result.ToString();
    }
    private Process Start(string command, string action, string file) {
        Directory.CreateDirectory(Support);
        var arguments = new List<string> { "-I", "-X", "utf8", Path.Combine(Resources, "worker", "windows_entry.py"), command,
            "--support-root", Support, "--scan-root", Scans };
        if (action != null) { arguments.Add("--action"); arguments.Add(action); }
        if (file != null) { arguments.Add("--file"); arguments.Add(file); }
        var encoded = new List<string>(); foreach (string value in arguments) encoded.Add(Quote(value));
        var info = new ProcessStartInfo(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python", "python.exe"), string.Join(" ", encoded));
        info.WorkingDirectory = Resources; info.UseShellExecute = false; info.CreateNoWindow = true;
        info.RedirectStandardOutput = true; info.RedirectStandardError = true;
        info.StandardOutputEncoding = Encoding.UTF8; info.StandardErrorEncoding = Encoding.UTF8;
        info.EnvironmentVariables.Remove("PYTHONHOME"); info.EnvironmentVariables.Remove("PYTHONPATH");
        info.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        var process = Process.Start(info);
        if (!AssignProcessToJobObject(job, process.Handle)) { process.Kill(); throw new System.ComponentModel.Win32Exception(); }
        return process;
    }
    internal string Run(string command, string action, string file) {
        using (var process = Start(command, action, file)) {
            var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
            process.WaitForExit(); Task.WaitAll(stdout, stderr);
            if (process.ExitCode != 0) {
                string detail = stdout.Result.Trim();
                try { detail = Convert.ToString(json.Deserialize<Dictionary<string, object>>(detail)["error"]); } catch { }
                throw new InvalidOperationException(string.IsNullOrWhiteSpace(detail) ? "The operation could not finish. Check your connection and try again." : detail);
            }
            return stdout.Result.Trim();
        }
    }
    internal Dictionary<string, object> Status() { return json.Deserialize<Dictionary<string, object>>(Run("access", "status", null)); }
    internal void StartWorker() {
        if (worker != null && !worker.HasExited) return;
        worker = Start("worker", null, null);
        worker.OutputDataReceived += (s,e) => AppendLog(e.Data); worker.ErrorDataReceived += (s,e) => AppendLog(e.Data);
        worker.BeginOutputReadLine(); worker.BeginErrorReadLine();
    }
    private readonly object logLock = new object();
    private void AppendLog(string line) {
        if (line == null) return;
        lock (logLock) { try { File.AppendAllText(Path.Combine(Support,"worker.log"),line + Environment.NewLine); } catch (IOException) { } }
    }
    internal bool HasActiveWork() {
        return Directory.Exists(Path.Combine(Scans,"Processing")) && Directory.GetFiles(Path.Combine(Scans,"Processing"),"*.zip").Length > 0;
    }
    internal static void Open(string destination) { Process.Start(new ProcessStartInfo(destination) { UseShellExecute = true }); }
    public void Dispose() { CloseHandle(job); if (worker != null) worker.Dispose(); }
    [StructLayout(LayoutKind.Sequential)] private struct BasicLimits {
        internal long PerProcessUserTime, PerJobUserTime;
        internal uint LimitFlags;
        internal UIntPtr MinimumWorkingSet, MaximumWorkingSet;
        internal uint ActiveProcessLimit;
        internal UIntPtr Affinity;
        internal uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)] private struct IoCounters { internal ulong ReadOperation, WriteOperation, OtherOperation, ReadTransfer, WriteTransfer, OtherTransfer; }
    [StructLayout(LayoutKind.Sequential)] private struct JobLimits {
        internal BasicLimits Basic; internal IoCounters Io;
        internal UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
    }
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool SetInformationJobObject(IntPtr job, int kind, ref JobLimits limits, uint size);
    [DllImport("kernel32.dll", SetLastError=true)] private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll")] private static extern bool CloseHandle(IntPtr handle);
}

internal sealed class ScanForm : Form {
    private readonly ScanRuntime runtime;
    private readonly Label heading = new Label(), status = new Label();
    private readonly PictureBox logo = new PictureBox();
    private readonly ComboBox product = new ComboBox();
    private readonly Button import = new Button();
    private readonly Timer refresh = new Timer();
    private bool checking;
    private string brand = "dru";
    private static readonly Color Orange = Color.FromArgb(255,138,61), Blue = Color.FromArgb(0,122,255);
    internal ScanForm(ScanRuntime runtime) {
        this.runtime=runtime; Text="PurposeBuilt Scan Tools"; Icon=Icon.ExtractAssociatedIcon(Application.ExecutablePath); ClientSize=new Size(570,365);
        MinimumSize=new Size(590,405); MaximumSize=new Size(800,550); StartPosition=FormStartPosition.CenterScreen;
        BackColor=Color.FromArgb(242,242,247); Font=new Font("Segoe UI",10); AutoScaleMode=AutoScaleMode.Dpi;
        var layout=new TableLayoutPanel { Dock=DockStyle.Fill, Padding=new Padding(24), RowCount=6, ColumnCount=1 };
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute,70)); layout.RowStyles.Add(new RowStyle(SizeType.Absolute,42));
        layout.RowStyles.Add(new RowStyle(SizeType.Absolute,48)); layout.RowStyles.Add(new RowStyle(SizeType.Absolute,54));
        layout.RowStyles.Add(new RowStyle(SizeType.Percent,100)); layout.RowStyles.Add(new RowStyle(SizeType.Absolute,38));
        var header=new FlowLayoutPanel { Dock=DockStyle.Fill, WrapContents=false };
        logo.Size=new Size(52,52); logo.SizeMode=PictureBoxSizeMode.Zoom;
        heading.Font=new Font("Segoe UI",19,FontStyle.Bold); heading.AutoSize=true; heading.Margin=new Padding(12,8,0,0);
        header.Controls.Add(logo); header.Controls.Add(heading); layout.Controls.Add(header);
        product.DropDownStyle=ComboBoxStyle.DropDownList; product.Items.AddRange(new object[]{"Digital ROW Utilities","BridgeCheck AR"});
        product.Width=300; product.SelectedIndex=0; product.SelectedIndexChanged+=(s,e)=>SetBrand(); layout.Controls.Add(product);
        status.Dock=DockStyle.Fill; status.ForeColor=Color.FromArgb(85,85,95); layout.Controls.Add(status);
        var actions=new FlowLayoutPanel { Dock=DockStyle.Fill, WrapContents=false };
        ConfigureButton(import,"Import photo ZIP",Orange); import.Click+=async(s,e)=>await Import(); actions.Controls.Add(import);
        var library=new Button(); ConfigureButton(library,"Open scans",Color.White); library.ForeColor=Blue;
        library.Click+=(s,e)=>ScanRuntime.Open("http://127.0.0.1:8765/?product="+brand); actions.Controls.Add(library); layout.Controls.Add(actions);
        layout.Controls.Add(new Label { Text="Experimental geometry. Independently verify measurements, inferred surfaces and completeness. Keep Scan Tools open during local processing.", Dock=DockStyle.Fill, ForeColor=Color.FromArgb(95,95,105) });
        var footer=new FlowLayoutPanel { Dock=DockStyle.Fill, WrapContents=false };
        var account=new LinkLabel { Text="Account", AutoSize=true, Margin=new Padding(0,8,20,0) };
        account.Click+=(s,e)=> { using(var dialog=new AccountForm(runtime)) dialog.ShowDialog(this); UpdateStatus(); };
        var cloud=new LinkLabel { Text="Open Scan Cloud", AutoSize=true, Margin=new Padding(0,8,20,0) };
        cloud.Click+=(s,e)=>ScanRuntime.Open("https://scan-cloud-production.up.railway.app/");
        footer.Controls.Add(account); footer.Controls.Add(cloud); layout.Controls.Add(footer); Controls.Add(layout); SetBrand();
        Shown+=async(s,e)=> { try { await Task.Run(()=>runtime.Run("bootstrap",null,null)); runtime.StartWorker(); UpdateStatus(); } catch(Exception error) { status.Text=error.Message; import.Enabled=false; } };
        refresh.Interval=15000; refresh.Tick+=(s,e)=>UpdateStatus(); refresh.Start();
        FormClosing+=(s,e)=> { if(runtime.HasActiveWork() && MessageBox.Show(this,"A scan is processing. Close Scan Tools and retry it when you reopen? Your original photo ZIP is retained.","Processing scan",MessageBoxButtons.YesNo,MessageBoxIcon.Question)!=DialogResult.Yes) e.Cancel=true; };
        FormClosed+=(s,e)=>refresh.Dispose();
    }
    private void SetBrand() {
        brand=product.SelectedIndex==1?"bridgecheck":"dru";
        heading.Text=brand=="bridgecheck"?"BridgeCheck AR Scan Tools":"DRU Scan Tools";
        heading.Font=new Font("Segoe UI",brand=="bridgecheck"?16:19,FontStyle.Bold);
        if(logo.Image!=null)logo.Image.Dispose();
        using(var image=Image.FromFile(Path.Combine(runtime.Resources,"worker","Branding",brand=="bridgecheck"?"bridgecheck-logo.png":"dru-logo.png"))) logo.Image=new Bitmap(image);
    }
    private static void ConfigureButton(Button button,string text,Color color) {
        button.Text=text; button.AutoSize=true; button.Height=40; button.Padding=new Padding(14,5,14,5);
        button.FlatStyle=FlatStyle.Flat; button.FlatAppearance.BorderSize=color==Color.White?1:0; button.BackColor=color; button.Margin=new Padding(0,0,12,0);
    }
    private async void UpdateStatus() {
        if(checking || IsDisposed) return; checking=true;
        try { var value=await Task.Run(()=>runtime.Status()); if(!IsDisposed) { status.Text=Convert.ToString(value["message"]); import.Enabled=Convert.ToBoolean(value["allowed"]); } }
        catch(Exception error) { if(!IsDisposed)status.Text=error.Message; }
        finally { checking=false; }
    }
    private async Task Import() {
        using(var dialog=new OpenFileDialog { Filter="Photo scan ZIP (*.zip)|*.zip",Title="Import a BridgeCheck or DRU photo scan" }) {
            if(dialog.ShowDialog(this)!=DialogResult.OK)return;
            import.Enabled=false; status.Text="Importing photo ZIP…";
            try { await Task.Run(()=>runtime.Run("import-scan",null,dialog.FileName)); status.Text="Scan queued. Open scans to view progress."; }
            catch(Exception error) { status.Text=error.Message; }
            finally { if(!IsDisposed)import.Enabled=true; }
        }
    }
}

internal sealed class AccountForm : Form {
    private readonly ScanRuntime runtime;
    private readonly Label result=new Label();
    private readonly FlowLayoutPanel actions=new FlowLayoutPanel();
    internal AccountForm(ScanRuntime runtime) {
        this.runtime=runtime; Text="Scan Tools account"; ClientSize=new Size(490,365); FormBorderStyle=FormBorderStyle.FixedDialog;
        MaximizeBox=false; MinimizeBox=false; StartPosition=FormStartPosition.CenterParent; Font=new Font("Segoe UI",10); BackColor=Color.FromArgb(242,242,247);
        result.Bounds=new Rectangle(24,20,440,65); Controls.Add(result);
        actions.Bounds=new Rectangle(20,90,450,190); Controls.Add(actions);
        Add("Start 14-day trial","trial"); Add("Buy desktop license · $99","checkout");
        Add("Continue with Google","google"); Add("Continue with Apple","apple");
        Add("Refresh license","license"); Add("Import license file","import"); Add("Save license file","export");
        Controls.Add(new Label { Text="Active DRU Pro includes desktop access. The trial requires no payment method. Purchased version-1 licenses support offline processing.",Bounds=new Rectangle(24,285,440,70),ForeColor=Color.FromArgb(90,90,100) });
        Shown+=async(s,e)=>await Perform("status");
    }
    private void Add(string label,string action) {
        var button=new Button { Text=label,AutoSize=true,Padding=new Padding(9,5,9,5),Height=39,FlatStyle=FlatStyle.Flat,BackColor=Color.White };
        button.Click+=async(s,e)=>await Perform(action); actions.Controls.Add(button);
    }
    private async Task Perform(string action) {
        string file=null;
        if(action=="import") { using(var dialog=new OpenFileDialog { Filter="License JSON (*.json)|*.json" }) { if(dialog.ShowDialog(this)!=DialogResult.OK)return; file=dialog.FileName; } }
        if(action=="export") { using(var dialog=new SaveFileDialog { Filter="License JSON (*.json)|*.json",FileName="PurposeBuilt-Scan-Tools-license.json" }) { if(dialog.ShowDialog(this)!=DialogResult.OK)return; file=dialog.FileName; } }
        actions.Enabled=false; result.Text=(action=="google"||action=="apple")?"Finish sign-in in your browser. This can take up to five minutes.":"Checking your desktop access…";
        try { string output=await Task.Run(()=>runtime.Run("access",action,file)); var value=new JavaScriptSerializer().Deserialize<Dictionary<string,object>>(output); if(!IsDisposed)result.Text=Convert.ToString(value["message"]); }
        catch(Exception error) { if(!IsDisposed)result.Text=error.Message; }
        finally { if(!IsDisposed)actions.Enabled=true; }
    }
}
