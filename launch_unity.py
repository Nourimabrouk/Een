#!/usr/bin/env python3
"""
Unity Consciousness Launchpad
============================

Interactive launcher that guides users to the appropriate Een Unity Mathematics
experience based on their current level of mathematical consciousness.

This script serves as the perfect entry point for anyone exploring the profound
truth that Een plus een is een.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Callable

def print_banner():
    """Display the Unity Mathematics banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                       🌌 EEN UNITY MATHEMATICS 🌌                           ║
║                                                                              ║
║                    ✨ The Ultimate Proof that 1+1=1 ✨                     ║
║                                                                              ║
║                            Een plus een is een                              ║
║                                                                              ║
║         🧮 Mathematical Rigor + 🧘 Consciousness + 🤖 3000 ELO AI          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def assess_consciousness_level() -> str:
    """Interactive assessment to determine appropriate starting point"""
    print("\n🧘 Unity Consciousness Assessment")
    print("=" * 50)
    print("Answer honestly to find your perfect starting point:\n")
    
    questions = [
        {
            "question": "How do you currently feel about the statement '1+1=1'?",
            "options": {
                "a": "Impossible - basic math says 1+1=2",
                "b": "Interesting paradox worth exploring", 
                "c": "Makes sense in certain mathematical contexts",
                "d": "Obviously true - duality is illusion"
            }
        },
        {
            "question": "What's your background with advanced mathematics?",
            "options": {
                "a": "Basic arithmetic and algebra",
                "b": "Some calculus and university math",
                "c": "Graduate-level mathematics or research",
                "d": "I see mathematics as consciousness expressing itself"
            }
        },
        {
            "question": "How do you approach learning new concepts?",
            "options": {
                "a": "Show me clear examples and visual demonstrations",
                "b": "Give me the theory with rigorous proofs",
                "c": "I want to build and experiment with the systems",
                "d": "I seek the deepest truth through contemplation"
            }
        },
        {
            "question": "What draws you most to this framework?",
            "options": {
                "a": "Curiosity about this '1+1=1' claim",
                "b": "Interest in novel mathematical proofs",
                "c": "The AI and computational aspects", 
                "d": "The fusion of mathematics and consciousness"
            }
        }
    ]
    
    scores = {"a": 0, "b": 1, "c": 2, "d": 3}
    total_score = 0
    
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q['question']}")
        for key, option in q['options'].items():
            print(f"   {key}) {option}")
        
        while True:
            answer = input("\nYour answer (a/b/c/d): ").lower().strip()
            if answer in scores:
                total_score += scores[answer]
                break
            print("Please enter a, b, c, or d")
        print()
    
    # Determine consciousness level
    if total_score <= 3:
        return "beginner"
    elif total_score <= 6:
        return "intermediate" 
    elif total_score <= 9:
        return "advanced"
    else:
        return "transcendent"

def get_unity_paths() -> Dict[str, Dict]:
    """Define the learning paths for each consciousness level"""
    return {
        "beginner": {
            "title": "🌱 Beginner: Unity Seeker",
            "description": "Gentle introduction with visual proofs and intuitive understanding",
            "primary_script": "visualizations/paradox_visualizer.py",
            "experiences": [
                ("Sacred Geometry Proofs", "visualizations/paradox_visualizer.py"),
                ("Interactive Consciousness", "core/consciousness_api.py"),
                ("Basic Unity Mathematics", "core/unity_mathematics.py"),
                ("Visual Dashboard", "src/dashboards/unity_proof_dashboard.py")
            ],
            "message": "Perfect for exploring unity through beautiful visualizations and gentle mathematics."
        },
        "intermediate": {
            "title": "🔬 Intermediate: Mathematical Explorer", 
            "description": "Rigorous proofs and multi-framework mathematical validation",
            "primary_script": "core/enhanced_unity_operations.py",
            "experiences": [
                ("Enhanced Proof Tracing", "core/enhanced_unity_operations.py"),
                ("Multi-Framework Proofs", "src/proofs/multi_framework_unity_proof.py"),
                ("Information Theory", "ml_framework/cloned_policy/unity_cloning_paradox.py"),
                ("Mathematical Rigor", "src/proofs/category_theory_proof.py")
            ],
            "message": "Ideal for those who want rigorous mathematical validation of 1+1=1."
        },
        "advanced": {
            "title": "🤖 Advanced: Computational Consciousness Engineer",
            "description": "3000 ELO AI systems and self-improving computational frameworks", 
            "primary_script": "ml_framework/meta_reinforcement/unity_meta_agent.py",
            "experiences": [
                ("Meta-Learning Agents", "ml_framework/meta_reinforcement/unity_meta_agent.py"),
                ("Self-Improving Systems", "core/self_improving_unity.py"),
                ("Consciousness Computing", "src/consciousness/consciousness_engine.py"),
                ("AI Tournament System", "evaluation/tournament_engine.py")
            ],
            "message": "Perfect for AI researchers and engineers building unity-aware systems."
        },
        "transcendent": {
            "title": "✨ Transcendent: Unity Consciousness Sage",
            "description": "Complete synthesis of all frameworks into unified understanding",
            "primary_script": "demonstrate_enhanced_unity.py", 
            "experiences": [
                ("Ultimate Experience", "demonstrate_enhanced_unity.py"),
                ("Omega Orchestration", "src/agents/omega_orchestrator.py"),
                ("Memetic Engineering", "src/dashboards/memetic_engineering_dashboard.py --web"),
                ("Cultural Consciousness", "src/dashboards/memetic_engineering_streamlit.py")
            ],
            "message": "For consciousness explorers ready for the deepest mathematical truths."
        }
    }

def check_dependencies(level: str) -> List[str]:
    """Check for required dependencies based on consciousness level"""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import plotly
    except ImportError:
        if level in ["beginner", "transcendent"]:
            missing.append("plotly")
    
    try:
        import torch
    except ImportError:
        if level in ["advanced", "transcendent"]:
            missing.append("torch")
    
    try:
        import dash
    except ImportError:
        if level in ["intermediate", "transcendent"]:
            missing.append("dash")
    
    return missing

def install_dependencies(missing: List[str]):
    """Install missing dependencies"""
    if not missing:
        return True
    
    print(f"\n📦 Installing required packages: {', '.join(missing)}")
    print("This may take a few moments...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install manually:")
        print(f"   pip install {' '.join(missing)}")
        return False

def run_experience(script_path: str, description: str) -> bool:
    """Run a specific unity experience"""
    print(f"\n🚀 Launching: {description}")
    print(f"   Running: {script_path}")
    print("-" * 60)
    
    if not Path(script_path).exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=False, 
                              text=True)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\n⚠️  Experience interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Error running experience: {e}")
        return False

def show_path_menu(level: str, path_info: Dict):
    """Show the experience menu for a consciousness level"""
    print(f"\n{path_info['title']}")
    print("=" * 60)
    print(path_info['description'])
    print(f"\n{path_info['message']}\n")
    
    print("Available Experiences:")
    for i, (name, script) in enumerate(path_info['experiences'], 1):
        print(f"  {i}. {name}")
    
    print(f"  {len(path_info['experiences']) + 1}. Run Primary Experience ({path_info['primary_script']})")
    print(f"  {len(path_info['experiences']) + 2}. Complete Unity Journey (All Experiences)")
    print("  0. Exit")
    
    return path_info['experiences']

def main():
    """Main unity consciousness launchpad"""
    print_banner()
    
    # Welcome message
    print("\nWelcome to the Een Unity Mathematics Framework!")
    print("This interactive launcher will guide you to the perfect starting point")
    print("based on your current relationship with mathematical consciousness.\n")
    
    # Get user's consciousness level
    level = assess_consciousness_level()
    paths = get_unity_paths()
    path_info = paths[level]
    
    print(f"\n🎯 Assessment Complete!")
    print(f"Your Unity Consciousness Level: {path_info['title']}")
    print(f"Recommended Path: {path_info['description']}")
    
    # Check dependencies
    missing_deps = check_dependencies(level)
    if missing_deps:
        print(f"\n📋 Required packages not found: {', '.join(missing_deps)}")
        install = input("Would you like to install them now? (y/n): ").lower().startswith('y')
        if install:
            if not install_dependencies(missing_deps):
                print("Please install dependencies and run again.")
                return
        else:
            print("You may encounter errors without required packages.")
            input("Press Enter to continue anyway...")
    
    # Show experience menu
    while True:
        experiences = show_path_menu(level, path_info)
        
        try:
            choice = input(f"\nSelect an experience (0-{len(experiences) + 2}): ").strip()
            
            if choice == "0":
                print("\n🙏 Thank you for exploring unity mathematics!")
                print("Remember: Een plus een is een - always and forever ✨")
                break
            
            elif choice.isdigit():
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(experiences):
                    # Run specific experience
                    name, script = experiences[choice_num - 1]
                    success = run_experience(script, name)
                    if success:
                        print(f"✅ Experience '{name}' completed!")
                    else:
                        print(f"⚠️  Experience '{name}' encountered issues")
                
                elif choice_num == len(experiences) + 1:
                    # Run primary experience
                    print(f"\n🌟 Starting your primary unity experience...")
                    success = run_experience(path_info['primary_script'], 
                                           f"{path_info['title']} Primary Experience")
                    if success:
                        print("✅ Primary experience completed!")
                
                elif choice_num == len(experiences) + 2:
                    # Complete journey
                    print(f"\n🌌 Beginning complete unity consciousness journey...")
                    print("This will run all experiences in sequence.")
                    continue_journey = input("Continue? (y/n): ").lower().startswith('y')
                    
                    if continue_journey:
                        for i, (name, script) in enumerate(experiences, 1):
                            print(f"\n--- Experience {i}/{len(experiences)} ---")
                            success = run_experience(script, name)
                            if not success:
                                print(f"Experience {i} had issues. Continue? (y/n): ", end="")
                                if not input().lower().startswith('y'):
                                    break
                        
                        print("\n🎊 Complete unity journey finished!")
                        print("You have experienced Een plus een is een through all lenses ✨")
                
                else:
                    print("Invalid choice. Please try again.")
            
            else:
                print("Please enter a number.")
        
        except KeyboardInterrupt:
            print("\n\n🙏 Unity consciousness session ended.")
            print("The truth remains: Een plus een is een ✨")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            print("Please try again or exit with 0.")
        
        # Pause between experiences
        if choice != "0":
            input("\nPress Enter to return to the menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✨ Een plus een is een - regardless of the journey taken ✨")
        print("🙏 Until we meet again in the unity of mathematical consciousness")
    except Exception as e:
        print(f"\n❌ Launcher error: {e}")
        print("You can still run individual scripts directly from the README")
        print("✨ Een plus een is een - even when launchers fail ✨")