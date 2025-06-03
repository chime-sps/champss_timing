import os
import time
import traceback
import subprocess
import datetime
from ..utils.utils import utils
from ..utils.notification import notification
from ..utils.logger import logger
from ..datastores.database import database
from ..pipecore.checker import checker

class MonitoringReport:
    def __init__(self, results, logger=logger(), verbose=False):
        self.results = results
        self.report_tex = ""
        self.logger = logger
        self.verbose = verbose

        # Get number of warnings
        self.n_warnings = 0
        for i, psr_result in enumerate(results):
            results[i]["n_warnings"] = 0
            for ckr in psr_result["checker_results"]:
                for itm in psr_result["checker_results"][ckr]:
                    results[i]["n_warnings"] += psr_result["checker_results"][ckr][itm]["level"]
            self.n_warnings += results[i]["n_warnings"]

        # Sort by number of warnings
        self.results = sorted(self.results, key=lambda x: x["n_warnings"], reverse=True)

    def latex_figure(self, image_path, text="", size=1.0):
        # Get absolute path to the image
        image_path = os.path.abspath(image_path)

        # Check if the image file exists
        if not os.path.exists(image_path):
            return f"""
            \\begin{{framed}}
            \\centering
            \\textbf{{Image not found: {image_path}}}
            \\end{{framed}}
            """
        
        # Check if is an acceptable image format
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
            return f"""
            \\begin{{framed}}
            \\centering
            \\textbf{{Unsupported image format: {image_path}}}
            \\end{{framed}}
            """
        
        return f"""
        \\begin{{framed}}
            \\centering
            \\includegraphics[width={size}\\linewidth]{{{image_path}}}
            {text}
        \\end{{framed}}
        """
    
    def latex_text(self, text):
        """
        Convert text to LaTeX format.
        Args:
            text (str): The text to convert.
        Returns:
            str: The LaTeX formatted text.
        """
        text = text.replace("\\", "\\textbackslash")  # Escape backslashes
        text = text.replace("_", "\\textunderscore ")  # Escape underscores
        text = text.replace("%", "\\%")  # Escape percent signs
        text = text.replace("$", "\\$")  # Escape dollar signs
        text = text.replace("&", "\\&")  # Escape ampersands
        text = text.replace("#", "\\#")  # Escape hash signs
        text = text.replace("{", "\\{")  # Escape curly braces
        text = text.replace("}", "\\}")  # Escape curly braces
        text = text.replace("~", "\\textasciitilde")  # Escape tilde
        text = text.replace("^", "\\textasciicircum")  # Escape caret
        text = text.replace("\n", "\\\\\n")
        text = text.replace("-", "$-$")  # Escape hyphens

        return text

    def generate_psr_summary(self, results):
        tex = ""
        # tex += "{\n\\centering\n"

        # Get parameters from the results
        psr_id = results["psr_id"]
        psr_dir = results["psr_dir"]
        checker_results = results["checker_results"]
        
        # Add title
        tex += f"\\section*{{{psr_id}}}\n"
        tex += f"\\addcontentsline{{toc}}{{section}}{{{psr_id}}}\n"

        # Add main diagnostics
        tex += self.latex_figure(os.path.join(psr_dir, "champss_diagnostic.pdf"))

        # Add checker results
        for ckr, items in checker_results.items():
            for item, result in items.items():
                level = result["level"]
                message = result["message"]
                id = result["id"]

                # Skip level 0 items
                if level == 0:  
                    continue
                
                # Start minipage
                tex += "\\begin{minipage}[t]{0.49\\textwidth}\n"
                tex += f"\\subsection*{{{self.latex_text(id.upper())}}}\n"
                tex += f"\\addcontentsline{{toc}}{{subsection}}{{{self.latex_text(id.upper())}}}\n"

                # Get and remove duplicate attachments
                attachments = list(set(result["attachments"] + result["attachments_report_only"]))
                for attachment in attachments:
                    if "champss_diagnostic.pdf" in attachment:
                        continue
                    tex += self.latex_figure(attachment)

                # Add item text
                if level  == 1:  # Only include warnings
                    tex += "\\textcolor{orange}{"
                elif level == 2:  # Include errors
                    tex += "\\textcolor{red}{"
                # tex += f"\\textbf{{[{self.latex_text(ckr)}.{self.latex_text(item)}]}} "
                tex += "\\texttt{" + f"{self.latex_text(message)} (Level: {level})" + "}\n" 
                tex += "}\n\n"

                # End minipage
                tex += "\\end{minipage}\n"
                tex += "\\hspace{0.02\\textwidth}\n"  # Add vertical space between items

        return tex

    def generate_front_page(self):
        return """
        \\title{Daily Monitoring Report}
        \\author{CHAMPSS Timing Pipeline}
        \\date{\\today}
        \\maketitle
        \\begin{framed}       
        """ + self.generate_overview() + """
        \\end{framed}
        \\tableofcontents
        """
    
    def generate_package_list(self):
        return """
        \\usepackage{graphicx}
        \\usepackage{framed}
        \\usepackage[a3paper, margin=0.75in]{geometry}
        \\usepackage{multicol}
        \\usepackage{xcolor}
        \\usepackage{hyperref}
        \\hypersetup{
            colorlinks,
            citecolor=black,
            filecolor=black,
            linkcolor=black,
            urlcolor=black
        }
        """
    
    def generate_overview(self):
        """
        Generate the overview section of the report.
        This section includes the total number of pulsars, 
        total number of warnings, and a summary of the results.
        """

        tex = ""
        tex += f"Total Pulsars: {len(self.results)}\\hspace{{0.5cm}}\n"
        tex += f"Total Warnings: {self.n_warnings}\\hspace{{0.5cm}}\n"
        tex += f"Host: {os.uname().nodename}\\hspace{{0.5cm}}\n"
        tex += f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return tex
    
    def get_tex(self):
        # Beginning of the document
        tex = "\\documentclass[]{article}\n"
        tex += self.generate_package_list()
        tex += "\n\\begin{document}\n"

        # Front page
        tex += self.generate_front_page()


        # In case of no warnings
        if self.n_warnings == 0:
            tex += "\\begin{framed}\n"
            tex += "\\centering\n"
            tex += "\\textbf{No warnings found for any pulsar.}\\hspace{0.5cm}\n\n"
            tex += "\\textit{All pulsars that been checked are healthy and no issues have been detected.}\\hspace{0.5cm}\n"
            tex += "\\end{framed}\n"
        else:
            # Add each pulsar's summary
            for psr_result in self.results:
                if psr_result["n_warnings"] == 0:
                    continue  # Skip pulsars with no warnings
                
                # Generate summary
                tex += "\n\\clearpage\n"
                tex += self.generate_psr_summary(psr_result)


        # End of the document
        tex += "\\end{document}\n"

        return tex
    
    def generate(self, outfile, latex="pdflatex", temdir="/tmp"):
        """
        Generate the LaTeX document for the monitoring report.
        Args:
            outfile (str): Path to the output PDF file.
            temdir (str): Temporary directory to store generated files (default: /tmp).
        Returns:
            str: The LaTeX document as a string.
        """

        work_id = utils.get_rand_string()

        # Get the LaTeX document 
        tex = self.get_tex()
        # print(tex)
        # return

        # Save the LaTeX document to a file
        tex_file = os.path.join(temdir, f"monitoring_report__{work_id}.tex")
        with open(tex_file, "w") as f:
            f.write(tex)
        self.logger.debug(f"Generated LaTeX file: {tex_file}")

        # Compile the LaTeX document to PDF
        pdf_file = os.path.join(temdir, f"monitoring_report__{work_id}.pdf")
        for _ in range(2):  # Run pdflatex twice to resolve references
            cmd = f"{latex} -interaction=nonstopmode -halt-on-error -output-directory={temdir} {tex_file}"
            if not self.verbose:
                os.system(cmd + " > /dev/null 2>&1")
            else:
                self.logger.info(f"Running command: {cmd}")
                os.system(cmd)
        self.logger.debug(f"Compiled PDF file: {pdf_file}")
        
        # Clean up the temporary LaTeX file
        if not self.verbose:
            os.remove(tex_file)

        # Check if the PDF was created successfully
        if not os.path.exists(pdf_file):
            self.logger.error(f"Failed to create PDF: {pdf_file}")
            return False
        
        # Move the PDF to the specified output file
        os.rename(pdf_file, outfile)
        self.logger.debug(f"Generated PDF report: {outfile}")

        return True
    
class Monitoring:
    def __init__(self, psrdirs=[], noti_hdl=notification(), logger=logger(), verbose=False):
        """
        Initialize the Monitoring class.
        
        Parameters
        ----------
        psrdirs : list
            List of pulsar directories to monitor.
        logger : logger
            Logger object for logging messages.
        verbose : bool
            If True, print debug messages.
        """
        
        self.psrdirs = psrdirs
        self.noti_hdl = noti_hdl
        self.logger = logger
        self.verbose = verbose

    def add_psrdir(self, psrdir):
        """
        Add a pulsar directory to the monitoring list.
        
        Parameters
        ----------
        psrdir : str
            Path to the pulsar directory.
        """
        
        if psrdir not in self.psrdirs:
            self.psrdirs.append(psrdir)
            self.logger.info(f"Added pulsar directory: {psrdir}")
        else:
            self.logger.warning(f"Pulsar directory already exists: {psrdir}")

    def run_checkers(self, within_24h=True, report=True):
        """
        Run all available checkers on the pulsar directories and generate a report.
        Parameters
        ----------
        within_24h : bool
            If True, only run checkers for pulsars updated in the last 24 hours.
        report : bool
            If True, generate a monitoring report.
        Returns
        -------
        bool
            True if all checkers passed, False otherwise.
        """
        
        results = []
        all_checkers_passed = True

        # Run checkers
        for psrdir in self.psrdirs:
            # Get info
            source_db = psrdir + "/champss_timing.sqlite3.db"
            psr_id = os.path.basename(psrdir)

            # Run checker
            try:
                with database(source_db) as db_hdl:
                    # Skip if updated more than 24 hours ago
                    if within_24h:
                        if db_hdl.get_last_timing_info()["timestamp"] < time.time() - 43200:
                            self.logger.warning(f"Skipping {psrdir} as it has not been updated in the last 24 hours.")
                            continue

                    # Run the checker
                    checker_res = checker(
                        psr_dir=psrdir,
                        db_hdl=db_hdl,
                        noti_hdl=self.noti_hdl, 
                        psr_id=psr_id, 
                        logger=self.logger.copy()
                    ).check()
                    
                    # check if all checkers are passed
                    for checker_module in checker_res.keys():
                        for checker_key in checker_res[checker_module].keys():
                            if checker_res[checker_module][checker_key]["level"] > 0:
                                all_checkers_passed = False

                    # Append to the results
                    results.append({
                        "psr_id": psr_id,
                        "psr_dir": psrdir, 
                        "checker_results": checker_res
                    })
            except Exception as e:
                self.logger.error(f"Error while checking {source_db}: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.noti_hdl.send_urgent_message(f"Error while checking {source_db}: {str(e)}")
                self.noti_hdl.send_code(traceback.format_exc())
                all_checkers_passed = False

        # Generate report
        if report:
            try:
                # Create a random file name for the report
                report_file = f"/tmp/monitoring_report__{utils.get_rand_string()}.pdf"

                # Generate the monitoring report
                report = MonitoringReport(
                    results=results,
                    logger=self.logger.copy(),
                    verbose=self.verbose
                )
                report.generate(outfile=report_file)

                # Send report file
                self.noti_hdl.send_file(report_file)

                # Clean up
                if not self.verbose:
                    os.unlink(report_file)  # Remove the report file after sending
            except Exception as e:
                self.logger.error(f"Error generating monitoring report: {str(e)}")
                self.logger.error(traceback.format_exc())
                self.noti_hdl.send_urgent_message(f"Error while generating monitoring report: {str(e)}")
                self.noti_hdl.send_code(traceback.format_exc())

        return all_checkers_passed